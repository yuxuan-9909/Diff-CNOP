import numpy as np
import os
import math
import datetime
import torch
import matplotlib.pyplot as plt
from EAAC_v2 import EAAC
from utils import (date_to_day_index, cfg_EAAC_V2, remove_prefix,
                   get_data_from_date_idx_EAAC_V35, get_climatology_data,
                   get_ecmwf_data_from_date_idx)

plt.rcParams['font.family'] = 'Liberation Sans'

def create_linear_sponge_weight(H, W, sponge_width=2, edge_weight=0.1):
    """边缘衰减的空间权重 (H, W)"""
    weight = torch.ones(H, W)
    for i in range(H):
        for j in range(W):
            min_dist = min(i, H - 1 - i, j, W - 1 - j)
            if min_dist < sponge_width:
                weight[i, j] = edge_weight + (1.0 - edge_weight) * (min_dist / sponge_width)
    return weight

class DiffCNOP:
    def __init__(
        self,
        model,
        regional_data,
        device,
        initial_doy,                        # ← 用于解析初始 doy
        epsilon_regional=3.0,
        tau=1.0,
        perturb_regional_time_steps=None,
        perturb_regional_var_channels=None,
        target_output_channel=-9,
        target_time_steps=None,
        target_H=20,
        target_W=20,
        slice_lat=slice(20, 40),
        slice_lon=slice(35, 55),
        mask_path='/public/home/qinbo/EAAC_V3/data/process_minmax/mask_regional.npy',
    ):
        self.device  = device
        self.epsilon_regional = epsilon_regional
        self.tau     = tau
        self.perturb_regional_time_steps  = perturb_regional_time_steps
        self.perturb_regional_var_channels = perturb_regional_var_channels
        self.target_output_channel = target_output_channel
        self.target_time_steps     = target_time_steps
        self.slice_lat = slice_lat
        self.slice_lon = slice_lon

        self.initial_doy = initial_doy

        # 冻结预报模型
        self.model = model
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()

        # 准备输入数据  (1, T, C, H, W)
        self.regional_data = torch.from_numpy(regional_data.copy()).float().unsqueeze(0).to(device)

        # 加载 mask
        mask_data = np.load(mask_path, allow_pickle=True)[:, :, :70]
        self._build_mask(mask_data, regional_data.shape)

        # 空间权重
        self.spatial_weight = create_linear_sponge_weight(target_H, target_W).to(device)

        # 控制预报
        with torch.no_grad():
            self.control_forecast = self._iterative_forecast_batch(self.regional_data)
            self.control_target   = self._extract_target_batch(self.control_forecast)

        # 扰动形状（单样本）
        self.perturbation_shape = (
            len(perturb_regional_time_steps),
            len(perturb_regional_var_channels),
            regional_data.shape[2],
            regional_data.shape[3],
        )

        print(f"   Score-based CNOP Sampler initialized  (Batch mode)")
        print(f"   (doy={self.initial_doy})")
        print(f"   Perturbation shape : {self.perturbation_shape}")
        print(f"   Temperature τ      : {self.tau}")
        print(f"   Constraint ε       : {self.epsilon_regional}")

    # ── mask ────────────────────────────────────────────────────────

    def _build_mask(self, mask_data, regional_shape):
        mask_list = []
        for t in self.perturb_regional_time_steps:
            for c in self.perturb_regional_var_channels:
                if c >= 29:
                    ch_mask = mask_data[c - 29, :, :]
                else:
                    ch_mask = np.ones((66, 70), dtype=np.float32)
                mask_list.append(ch_mask)
        regional_mask = np.stack(mask_list).reshape(
            len(self.perturb_regional_time_steps),
            len(self.perturb_regional_var_channels),
            regional_shape[2], regional_shape[3],
        )
        self.regional_mask = torch.from_numpy(regional_mask).float().to(self.device)

    # ── iterative forecast ──────────────────────────────────────────

    def _iterative_forecast_batch(self, regional_input):
        """6步自回归预报.
        regional_input : (B, T_in=2, C, H, W)
        returns        : (B, 6, C, H, W)
        每一步 doy 向前推进 5 天（一个 pentad）。
        """
        B = regional_input.shape[0]
        current_x = regional_input[:, :2]
        all_preds = []

        for step in range(6):
            doy_val = (self.initial_doy + (step + 1) * 5 - 1) % 365 + 1
            doy = torch.full((B,), doy_val, dtype=torch.float32, device=self.device)

            x_pred = self.model(current_x, doy)          # (B, 1, C, H, W)
            all_preds.append(x_pred)
            current_x = torch.cat([current_x[:, 1:2], x_pred], dim=1)

        return torch.cat(all_preds, dim=1)               # (B, 6, C, H, W)

    def _extract_target_batch(self, forecast):
        """forecast: (B, 6, C, H, W) → (B, len(target_time_steps))"""
        target = forecast[:, self.target_time_steps, self.target_output_channel, :, :]
        target_crop = target[..., self.slice_lat, self.slice_lon]
        return (target_crop * self.spatial_weight).sum(dim=(-2, -1))

    def _apply_perturbation_batch(self, delta):
        """delta: (B, T_pert, C_pert, H, W) → perturbed input (B, T, C, H, W)"""
        B = delta.shape[0]
        perturbed = self.regional_data.expand(B, -1, -1, -1, -1).clone()
        for i, t in enumerate(self.perturb_regional_time_steps):
            for j, c in enumerate(self.perturb_regional_var_channels):
                mask = (self.regional_mask[i, j] > 0.5).unsqueeze(0).expand(B, -1, -1)
                base = self.regional_data[:, t, c].expand(B, -1, -1)
                perturbed[:, t, c] = torch.where(mask, base + delta[:, i, j], base)
        return perturbed

    def _project_to_constraint_batch(self, delta):
        """L2 球投影，保留 mask 区域内的范数约束"""
        B = delta.shape[0]
        masked = delta * self.regional_mask.unsqueeze(0)
        norm   = torch.norm(masked.view(B, -1), dim=1, keepdim=True)       # (B,1)
        scale  = torch.clamp(self.epsilon_regional / (norm + 1e-8), max=1.0)
        return masked * scale.view(B, 1, 1, 1, 1)

    def compute_objective_batch(self, delta):
        """J(δ) = Σ_t (f_t(x+δ) - f_t(x))²   shape: (B,)"""
        perturbed = self._apply_perturbation_batch(delta)
        forecast  = self._iterative_forecast_batch(perturbed)
        target    = self._extract_target_batch(forecast)
        diff      = target - self.control_target.expand(delta.shape[0], -1)
        return (diff ** 2).sum(dim=1)

    def compute_score_batch(self, delta):
        """∇_δ J / τ   (Langevin score)"""
        delta_var = delta.clone().detach().requires_grad_(True)
        J = self.compute_objective_batch(delta_var)
        J.sum().backward()
        score = delta_var.grad.clone() / self.tau
        return score, J.detach()


    def sample_cnop_batch(self, batch_size, sigma_levels,
                          steps_per_level=50, step_size=0.01,
                          init_scale=0.1, log_interval=10):
        """Annealed Langevin dynamics → batch of CNOP perturbations"""
        delta = torch.randn(batch_size, *self.perturbation_shape, device=self.device) * init_scale
        delta = self._project_to_constraint_batch(delta)

        total_steps = 0
        J_values = torch.zeros(batch_size, device=self.device)

        for level_idx, sigma in enumerate(sigma_levels):
            for step in range(steps_per_level):
                score, J_values = self.compute_score_batch(delta)
                noise = torch.randn_like(delta)
                delta = delta + (step_size / 2) * score + sigma * math.sqrt(step_size) * noise
                delta = self._project_to_constraint_batch(delta)
                total_steps += 1

                if total_steps % log_interval == 0:
                    norms = torch.norm(
                        (delta * self.regional_mask.unsqueeze(0)).view(batch_size, -1), dim=1)
                    print(f"   Level {level_idx+1}/{len(sigma_levels)}, "
                          f"Step {step+1}/{steps_per_level} | "
                          f"J: {J_values.mean():.4f} ± {J_values.std():.4f} | "
                          f"σ: {sigma:.4f} | ‖δ‖: {norms.mean():.4f}")

        final_norms = torch.norm(
            (delta * self.regional_mask.unsqueeze(0)).view(batch_size, -1), dim=1)
        info = {
            'final_J':    J_values.cpu().numpy(),
            'final_norm': final_norms.cpu().numpy(),
            'total_steps': total_steps,
        }
        return delta.detach(), info


    def sample_ensemble(self, n_members, save_dir, hindcast_date=None, batch_size=10,
                        sigma_max=0.5, sigma_min=0.0001, num_levels=20,
                        steps_per_level=100, step_size=0.1, init_scale=0.5):
        """对称守恒集合生成：优化成员 + 取反成员交替保存"""
        sigma_levels = np.geomspace(sigma_max, sigma_min, num_levels)
        os.makedirs(save_dir, exist_ok=True)

        n_optimized = (n_members + 1) // 2
        n_batches   = (n_optimized + batch_size - 1) // batch_size

        print(f"   对称守恒集合生成模式")
        print(f"   总成员数: {n_members}  (优化: {n_optimized}, 取反: {n_members - n_optimized})")

        member_id      = 0
        optimized_count = 0

        for batch_idx in range(n_batches):
            cur_bs = min(batch_size, n_optimized - optimized_count)

            print(f"Batch {batch_idx+1}/{n_batches}  (优化 {cur_bs} 个成员)")

            delta_batch, info = self.sample_cnop_batch(
                batch_size=cur_bs,
                sigma_levels=sigma_levels,
                steps_per_level=steps_per_level,
                step_size=step_size,
                init_scale=init_scale,
                log_interval=steps_per_level * 2,
            )

            # 取反成员
            delta_neg_batch = self._project_to_constraint_batch(-delta_batch)

            with torch.no_grad():
                J_neg     = self.compute_objective_batch(delta_neg_batch)
                norms_neg = torch.norm(
                    (delta_neg_batch * self.regional_mask.unsqueeze(0)).view(cur_bs, -1), dim=1)

                # 批量预报（正 & 负）
                forecast_pos = self._iterative_forecast_batch(
                    self._apply_perturbation_batch(delta_batch))
                forecast_neg = self._iterative_forecast_batch(
                    self._apply_perturbation_batch(delta_neg_batch))

            for i in range(cur_bs):
                if member_id >= n_members:
                    break

                # 优化成员
                self.save_single_member(
                    member_id, delta_batch[i:i+1], forecast_pos[i:i+1],
                    save_dir, hindcast_date, is_negated=False)
                print(f"Member {member_id:03d} [优化]: "
                      f"J={info['final_J'][i]:.4f}, norm={info['final_norm'][i]:.4f}")
                member_id += 1

                if member_id >= n_members:
                    break

                # 取反成员
                self.save_single_member(
                    member_id, delta_neg_batch[i:i+1], forecast_neg[i:i+1],
                    save_dir, hindcast_date, is_negated=True)
                print(f"Member {member_id:03d} [取反]: "
                      f"J={J_neg[i].item():.4f}, norm={norms_neg[i].item():.4f}")
                member_id += 1

            optimized_count += cur_bs

        print(f"All {n_members} members saved to {save_dir}")

        self.generate_summary(save_dir, n_members, hindcast_date)
        self._plot_ensemble_overview(save_dir, hindcast_date)

    # ── save / visualize ────────────────────────────────────────────

    def save_single_member(self, member_id, delta, forecast, save_dir,
                           hindcast_date=None, is_negated=False):
        os.makedirs(save_dir, exist_ok=True)
        delta_np    = delta.cpu().numpy()    if torch.is_tensor(delta)    else delta
        forecast_np = forecast.cpu().numpy() if torch.is_tensor(forecast) else forecast

        results = dict(
            member_id    = member_id,
            delta        = delta_np,          # (1, T_pert, C_pert, H, W)
            forecast     = forecast_np,       # (1, 6, C, H, W)
            control_forecast = self.control_forecast.cpu().numpy(),
            tau          = self.tau,
            epsilon_regional = self.epsilon_regional,
            perturb_regional_time_steps  = self.perturb_regional_time_steps,
            perturb_regional_var_channels = self.perturb_regional_var_channels,
            regional_mask = self.regional_mask.cpu().numpy(),
            is_negated   = is_negated,
        )
        if hindcast_date is not None:
            results['hindcast_date'] = hindcast_date

        np.savez(os.path.join(save_dir, f"member_{member_id:03d}.npz"), **results)
        self._plot_member_vs_control(
            member_id, forecast_np, self.control_forecast.cpu().numpy(),
            save_dir, hindcast_date, is_negated)

    def _plot_member_vs_control(self, member_id, member_forecast, control_forecast,
                                save_dir, hindcast_date=None, is_negated=False):
        from matplotlib.gridspec import GridSpec

        ch = self.target_output_channel
        var_min, var_max = 212.09735, 316.29013

        control = control_forecast[0, :, ch] * (var_max - var_min) + var_min
        member  = member_forecast[0,  :, ch] * (var_max - var_min) + var_min

        if hindcast_date is not None:
            clima = get_climatology_data(hindcast_date)[2:8]
            control -= clima
            member  -= clima

        v_abs = max(abs(control.min()), abs(control.max()),
                    abs(member.min()),  abs(member.max()))
        label = "[取反]" if is_negated else "[优化]"

        fig = plt.figure(figsize=(25, 8))
        gs  = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.25)

        for t in range(6):
            ax = fig.add_subplot(gs[0, t])
            im = ax.imshow(control[t], cmap='RdBu_r', vmin=-v_abs, vmax=v_abs, aspect='auto')
            ax.set_title(f'Control T+{t+1}', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel('Control Forecast', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for t in range(6):
            ax = fig.add_subplot(gs[1, t])
            im = ax.imshow(member[t], cmap='RdBu_r', vmin=-v_abs, vmax=v_abs, aspect='auto')
            ax.set_title(f'Member {member_id:03d} T+{t+1}', fontsize=11, fontweight='bold')
            ax.set_xticks([]); ax.set_yticks([])
            if t == 0:
                ax.set_ylabel(f'Member {member_id:03d} {label}', fontsize=12, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        title = f'Control vs Member {member_id:03d} {label} - Channel {ch}'
        if hindcast_date:
            title += f' - {hindcast_date}'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        plt.savefig(os.path.join(save_dir, f"member_{member_id:03d}_vs_control.png"),
                    dpi=150, bbox_inches='tight')
        plt.close()

    def generate_summary(self, save_dir, n_members, hindcast_date=None):
        print(f"\nGenerating summary file...")

        ensemble_deltas    = []
        ensemble_forecasts = []
        is_negated_list    = []

        for i in range(n_members):
            d = np.load(os.path.join(save_dir, f"member_{i:03d}.npz"))
            ensemble_deltas.append(d['delta'])
            ensemble_forecasts.append(d['forecast'])
            is_negated_list.append(d['is_negated'])

        ensemble_deltas    = np.array(ensemble_deltas)    # (N, 1, T_pert, C_pert, H, W)
        ensemble_forecasts = np.array(ensemble_forecasts) # (N, 1, 6, C, H, W)
        is_negated_array   = np.array(is_negated_list)

        results = dict(
            ensemble_deltas    = ensemble_deltas,
            ensemble_forecasts = ensemble_forecasts,
            control_forecast   = self.control_forecast.cpu().numpy(),
            mu_regional        = ensemble_deltas.mean(axis=0),
            std_regional       = ensemble_deltas.std(axis=0),
            n_members          = n_members,
            tau                = self.tau,
            epsilon_regional   = self.epsilon_regional,
            perturb_regional_time_steps  = self.perturb_regional_time_steps,
            perturb_regional_var_channels = self.perturb_regional_var_channels,
            regional_mask      = self.regional_mask.cpu().numpy(),
            is_negated         = is_negated_array,
        )
        if hindcast_date is not None:
            results['hindcast_date'] = hindcast_date

        summary_path = os.path.join(save_dir, "ensemble_summary.npz")
        np.savez(summary_path, **results)
        print(f"Summary saved to {summary_path}")
        print(f"优化成员: {(~is_negated_array).sum()}, "
              f"取反成员: {is_negated_array.sum()}")

        if hindcast_date is not None:
            self._run_visualizations(summary_path, save_dir, hindcast_date)

    def _run_visualizations(self, summary_path, save_dir, hindcast_date):
        try:
            from plot_scripts import (
                compare_ensemble_with_control,
                visualize_variational_parameters,
            )
            compare_ensemble_with_control(
                summary_path, hindcast_date, save_dir, self.target_output_channel)
            visualize_variational_parameters(summary_path, hindcast_date, save_dir)
            print(f"All visualizations saved to {save_dir}")
        except Exception as e:
            print(f"Visualization failed: {e}")

    def _plot_ensemble_overview(self, save_dir, hindcast_date):
        data = np.load(os.path.join(save_dir, "ensemble_summary.npz"), allow_pickle=True)

        ensemble_forecasts = data['ensemble_forecasts']   # (N, 1, 6, C, H, W)
        control_forecast   = data['control_forecast']     # (1, 6, C, H, W)
        is_negated         = data['is_negated']           # (N,)

        clima_data = get_climatology_data(hindcast_date)[2:]
        _, _, truth = get_ecmwf_data_from_date_idx(hindcast_date)
        truth = truth - clima_data

        ch = 25
        var_min, var_max = 212.09735, 316.29013

        ens_ch  = ensemble_forecasts[:, 0, :, ch]             # (N, 6, H, W)
        ctrl_ch = control_forecast[0, :, ch]                  # (6, H, W)

        ens_anom  = ens_ch  * (var_max - var_min) + var_min - clima_data
        ctrl_anom = ctrl_ch * (var_max - var_min) + var_min - clima_data

        region_config = {
            '20210517': (33, 44, 40, 51), '20210829': (35, 42, 40, 51),
            '20210905': (35, 42, 40, 51), '20220525': (27, 34, 40, 51),
            '20220617': (33, 38, 32, 40), '20220721': (31, 40, 35, 50),
            '20220817': (15, 24, 40, 51), '20230525': (18, 25, 40, 51),
            '20230613': (27, 34, 40, 51), '20230725': (27, 34, 40, 51),
            '20240517': (27, 34, 40, 51), '20240801': (30, 40, 32, 40),
            '20240813': (30, 40, 32, 40),
        }
        r = region_config.get(hindcast_date, (27, 34, 40, 51))
        sl = (slice(None), slice(r[0], r[1]), slice(r[2], r[3]))

        ens_reg   = ens_anom[sl[0],  :, sl[1], sl[2]].mean(axis=(-2, -1))   # (N, 6) — wait, wrong indexing
        # fix: ens_anom is (N,6,H,W)
        ens_reg   = ens_anom[:, :, r[0]:r[1], r[2]:r[3]].mean(axis=(-2, -1))  # (N, 6)
        ctrl_reg  = ctrl_anom[:, r[0]:r[1], r[2]:r[3]].mean(axis=(-2, -1))    # (6,)
        truth_reg = truth[:6, r[0]:r[1], r[2]:r[3]].mean(axis=(-2, -1))        # (6,)

        ens_mean = ens_reg.mean(axis=0)
        n_members = ens_reg.shape[0]
        optimized_idx = ~is_negated

        time_steps = [1, 2, 3, 4, 5, 6]
        plt.figure(figsize=(10, 7))

        for i, idx in enumerate(np.where(optimized_idx)[0]):
            plt.plot(time_steps, ens_reg[idx], 'steelblue', linestyle='--',
                     linewidth=0.8, alpha=0.5, label='Optimized members' if i == 0 else None)
        for i, idx in enumerate(np.where(is_negated)[0]):
            plt.plot(time_steps, ens_reg[idx], 'darkorange', linestyle='--',
                     linewidth=0.8, alpha=0.5, label='Negated members' if i == 0 else None)

        plt.plot(time_steps, ens_mean,  'b-', lw=3, marker='o', ms=8,  label='Ensemble Mean')
        plt.plot(time_steps, ctrl_reg,  'g-', lw=3, marker='s', ms=8,  label='Control')
        plt.plot(time_steps, truth_reg, 'r-', lw=3, marker='*', ms=10, label='Truth (ERA5)')

        plt.xlabel('Lead Time (pentads)', fontsize=12)
        plt.ylabel('Temperature Anomaly (K)', fontsize=12)
        plt.title(
            f'Score-based Ensemble Forecast (Symmetric) – {hindcast_date}\n'
            f'(Total: {n_members},  Optimized: {optimized_idx.sum()},  '
            f'Negated: {is_negated.sum()})',
            fontsize=14, fontweight='bold')
        plt.xticks(time_steps)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        fig_path = os.path.join(save_dir, 'ensemble_overview.png')
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Ensemble overview saved to {fig_path}")


if __name__ == '__main__':

    all_cases_config = {
        '20210517': {'target_H': 15, 'target_W': 20, 'slice_lat': slice(30, 45), 'slice_lon': slice(35, 55)},
        '20210829': {'target_H': 20, 'target_W': 20, 'slice_lat': slice(25, 45), 'slice_lon': slice(33, 53)},
        '20210905': {'target_H': 20, 'target_W': 25, 'slice_lat': slice(25, 45), 'slice_lon': slice(30, 55)},
        # '20220525': {'target_H': 20, 'target_W': 20, 'slice_lat': slice(20, 40), 'slice_lon': slice(35, 55)},
        # '20220617': {'target_H': 15, 'target_W': 20, 'slice_lat': slice(25, 40), 'slice_lon': slice(25, 45)},
        # '20220721': {'target_H': 20, 'target_W': 25, 'slice_lat': slice(25, 45), 'slice_lon': slice(30, 55)},
        # '20230525': {'target_H': 20, 'target_W': 20, 'slice_lat': slice(10, 30), 'slice_lon': slice(35, 55)},
        # '20230613': {'target_H': 20, 'target_W': 20, 'slice_lat': slice(20, 40), 'slice_lon': slice(35, 55)},
        # '20230725': {'target_H': 15, 'target_W': 20, 'slice_lat': slice(25, 40), 'slice_lon': slice(35, 55)},
        # '20240801': {'target_H': 20, 'target_W': 20, 'slice_lat': slice(25, 45), 'slice_lon': slice(30, 50)},
    }

    tau             = 0.5
    model_ckpt_path = '/public/home/qinbo/EAAC_V3/lightning_logs/version_2694564/checkpoints/exp06-epoch=15-val_loss=0.000443.ckpt'
    save_base_dir   = (f'/public/home/qinbo/EAAC_V3/results/diff_dm_3_symmetry/'
                       f'cnop_diff_cases_explanation_tau{tau}')
    num_members     = 50
    batch_size      = 10

    sampling_config = dict(
        sigma_max=0.5, sigma_min=0.005, num_levels=20,
        steps_per_level=50, step_size=0.1, init_scale=0.5,
    )

    device = torch.device('cuda:0')
    print(f"Using device: {device}")

    model = EAAC(**cfg_EAAC_V2)
    state = torch.load(model_ckpt_path, map_location=device)['state_dict']
    model.load_state_dict(remove_prefix(state, prefix='model.'))
    model = model.to(device).eval()
    print("Forecast model loaded")

    for hindcast_date, case_config in all_cases_config.items():
        print(f'# Processing {hindcast_date}')

        save_dir       = os.path.join(save_base_dir, hindcast_date)
        date_idx       = date_to_day_index(hindcast_date)
        regional_sample, doy_sample = get_data_from_date_idx_EAAC_V35(date_idx)

        cnop_sampler = DiffCNOP(
            model          = model,
            regional_data  = regional_sample,
            device         = device,
          #   hindcast_date  = hindcast_date,
            initial_doy = float(doy_sample[1]),
            epsilon_regional = 3.0,
            tau            = tau,
            perturb_regional_time_steps  = [1],
            perturb_regional_var_channels = (list(range(0, 20)) + list(range(25, 26)) + list(range(29, 34))),
            target_output_channel = -9,
            target_time_steps     = [4, 5],
            target_H = case_config['target_H'],
            target_W = case_config['target_W'],
            slice_lat = case_config['slice_lat'],
            slice_lon = case_config['slice_lon'],
            mask_path = '/public/home/qinbo/EAAC_V3/data/process_minmax/mask_regional.npy',
        )

        cnop_sampler.sample_ensemble(
            n_members     = num_members,
            save_dir      = save_dir,
            hindcast_date = hindcast_date,
            batch_size    = batch_size,
            **sampling_config,
        )

        print(f"\n{hindcast_date} completed!")

    print("All cases completed!")