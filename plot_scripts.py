import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os


def compare_ensemble_with_control(
    npz_path,
    hindcast_date,
    save_dir,
    channel_idx=-9,
):
    """控制预报 vs 集合平均对比图（2行×6列）"""
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)
    ensemble_forecasts = data['ensemble_forecasts'][:, 0, :, channel_idx, :, :]  # (N, 6, H, W)
    control_forecast   = data['control_forecast'][0, :, channel_idx, :, :]       # (6, H, W)

    from utils import get_climatology_data
    clima = get_climatology_data(hindcast_date)[2:8]   # (6, H, W)

    var_min, var_max = 212.09735, 316.29013
    ens_anom  = ensemble_forecasts * (var_max - var_min) + var_min - clima[np.newaxis]
    ctrl_anom = control_forecast   * (var_max - var_min) + var_min - clima

    ens_mean = ens_anom.mean(axis=0)   # (6, H, W)

    v_abs = max(abs(ens_mean.min()),   abs(ens_mean.max()),
                abs(ctrl_anom.min()),  abs(ctrl_anom.max()))

    fig = plt.figure(figsize=(25, 8))
    gs  = GridSpec(2, 6, figure=fig, hspace=0.3, wspace=0.25)

    for t in range(6):
        ax = fig.add_subplot(gs[0, t])
        im = ax.imshow(ctrl_anom[t], cmap='RdBu_r', vmin=-v_abs, vmax=v_abs,
                       aspect='auto', interpolation='nearest')
        ax.set_title(f'Control T+{t+1}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Control Forecast', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    for t in range(6):
        ax = fig.add_subplot(gs[1, t])
        im = ax.imshow(ens_mean[t], cmap='RdBu_r', vmin=-v_abs, vmax=v_abs,
                       aspect='auto', interpolation='nearest')
        ax.set_title(f'Ensemble Mean T+{t+1}', fontsize=11, fontweight='bold')
        ax.set_xticks([]); ax.set_yticks([])
        if t == 0:
            ax.set_ylabel('Ensemble Mean', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f'Control vs Ensemble Mean  –  Channel {channel_idx}  –  {hindcast_date}',
        fontsize=16, fontweight='bold', y=0.995)

    plt.savefig(
        os.path.join(save_dir, f'control_vs_ensemble_ch{channel_idx}_{hindcast_date}.png'),
        dpi=150, bbox_inches='tight')
    plt.close()


def visualize_variational_parameters(
    npz_path,
    hindcast_date,
    save_dir,
):
    """为每个被扰动的变量绘制均值 (μ) 和标准差 (σ) 的空间分布图。"""
    os.makedirs(save_dir, exist_ok=True)

    data = np.load(npz_path, allow_pickle=True)

    required = ['mu_regional', 'std_regional',
                'perturb_regional_var_channels',
                'perturb_regional_time_steps', 'regional_mask']
    missing = [k for k in required if k not in data]
    if missing:
        print(f"Missing keys in npz: {missing}")
        return

    mu_regional  = data['mu_regional']               # (1, n_time, n_vars, H, W)
    std_regional = data['std_regional']               # (1, n_time, n_vars, H, W)
    var_channels = data['perturb_regional_var_channels']
    time_steps   = data['perturb_regional_time_steps']
    regional_mask = data['regional_mask']             # (1, n_time, n_vars, H, W)  or (n_time, n_vars, H, W)

    # ── variable name lookup (EAAC-S2S v2 channel layout) ──
    var_names = {
        **{i:    f'Q_{p}hPa'  for i, p in enumerate([200,500,700,850,1000])},
        **{i+5:  f'T_{p}hPa'  for i, p in enumerate([200,500,700,850,1000])},
        **{i+10: f'U_{p}hPa'  for i, p in enumerate([200,500,700,850,1000])},
        **{i+15: f'V_{p}hPa'  for i, p in enumerate([200,500,700,850,1000])},
        **{i+20: f'Z_{p}hPa'  for i, p in enumerate([200,500,700,850,1000])},
        25: 'T2M', 26: 'MSLP', 27: 'U10', 28: 'V10',
        29: 'SST', 30: 'STL1', 31: 'STL2', 32: 'SWVL1', 33: 'SWVL2',
    }

    n_times = len(time_steps)

    for var_idx, var_ch in enumerate(var_channels):
        var_name = var_names.get(int(var_ch), f'Var{var_ch}')

        fig, axes = plt.subplots(n_times, 2, figsize=(14, 5 * n_times))
        if n_times == 1:
            axes = axes.reshape(1, -1)

        for time_idx, time_step in enumerate(time_steps):
            mu_sl   = mu_regional [0, time_idx, var_idx]   # (H, W)
            std_sl  = std_regional[0, time_idx, var_idx]
            mask_sl = regional_mask[0, time_idx, var_idx] if regional_mask.ndim == 5 \
                      else regional_mask[time_idx, var_idx]

            valid = mask_sl > 0.5
            mu_masked  = np.where(valid, mu_sl,  np.nan)
            std_masked = np.where(valid, std_sl, np.nan)

            mu_valid  = mu_sl [valid]
            std_valid = std_sl[valid]

            # ── left: μ ──
            ax = axes[time_idx, 0]
            v_abs = max(abs(mu_valid.min()), abs(mu_valid.max()))
            im = ax.imshow(mu_masked, cmap='RdBu_r', vmin=-v_abs, vmax=v_abs,
                           aspect='auto', interpolation='nearest')
            ax.set_title(
                f'μ  –  Time Step {time_step}\n'
                f'mean={mu_valid.mean():.4f}  std={mu_valid.std():.4f}  '
                f'range=[{mu_valid.min():.4f}, {mu_valid.max():.4f}]',
                fontsize=10, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude',  fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                'μ', rotation=270, labelpad=12, fontsize=9)

            # ── right: σ ──
            ax = axes[time_idx, 1]
            im = ax.imshow(std_masked, cmap='YlOrRd', vmin=0, vmax=std_valid.max(),
                           aspect='auto', interpolation='nearest')
            ax.set_title(
                f'σ  –  Time Step {time_step}\n'
                f'mean={std_valid.mean():.4f}  std={std_valid.std():.4f}  '
                f'range=[{std_valid.min():.4f}, {std_valid.max():.4f}]',
                fontsize=10, fontweight='bold')
            ax.set_xlabel('Longitude', fontsize=9)
            ax.set_ylabel('Latitude',  fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label(
                'σ', rotation=270, labelpad=12, fontsize=9)

        fig.suptitle(
            f'CNOP Perturbation Parameters  –  {var_name} (ch {var_ch})  –  {hindcast_date}',
            fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()

        plt.savefig(
            os.path.join(save_dir, f'cnop_params_{var_name}_ch{var_ch}_{hindcast_date}.png'),
            dpi=150, bbox_inches='tight')
        plt.close()