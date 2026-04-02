import numpy as np
from datetime import datetime, timedelta
import os

def date_to_day_index(date_string):
    """'YYYYMMDD' → days from 2020-01-01 (1-based)."""
    base  = datetime(2020, 1, 1)
    delta = datetime.strptime(date_string, '%Y%m%d') - base
    return delta.days + 1


def remove_prefix(state_dict, prefix='model.'):
    """Remove a key prefix from a state_dict (e.g. from Lightning checkpoints)."""
    return {
        (k[len(prefix):] if k.startswith(prefix) else k): v
        for k, v in state_dict.items()
    }

def get_data_from_date_idx_EAAC_V35(date_idx, doy_path='/public/home/qinbo/EAAC_V3/data/process_minmax/doy_test.npy'):

    regional_3d_vars = ['q', 't', 'u', 'v', 'z']
    regional_2d_vars = ['t2m', 'msl', 'u10', 'v10']
    boundary_vars    = ['sst', 'stl1', 'stl2', 'swvl1', 'swvl2']

    data_root = '/public/home/qinbo/EAAC_V3/data/process_minmax'
    window    = slice(date_idx - 10, date_idx + 30)   # 40 days → 8 pentads

    regional_sample = []

    for var in regional_3d_vars:
        raw = np.memmap(
            f'{data_root}/{var}_test.memmap',
            dtype='float32', shape=(1827, 8, 66, 71)
        )[window, [1, 4, 5, 6, 7], :, :70]            # (40, 5, 66, 70)
        pentad = raw.reshape(8, 5, *raw.shape[1:]).mean(axis=1)  # (8, 5, 66, 70)
        regional_sample.append(pentad)

    for var in regional_2d_vars:
        raw = np.memmap(
            f'{data_root}/{var}_test.memmap',
            dtype='float32', shape=(1827, 66, 71)
        )[window, :, :70]                              # (40, 66, 70)
        pentad = raw.reshape(8, 5, *raw.shape[1:]).mean(axis=1)  # (8, 66, 70)
        regional_sample.append(pentad[:, np.newaxis])  # (8, 1, 66, 70)

    for var in boundary_vars:
        raw = np.memmap(
            f'{data_root}/{var}_test.memmap',
            dtype='float32', shape=(1827, 66, 71)
        )[window, :, :70]                              # (40, 66, 70)
        pentad = raw.reshape(8, 5, *raw.shape[1:]).mean(axis=1)  # (8, 66, 70)
        regional_sample.append(pentad[:, np.newaxis])  # (8, 1, 66, 70)

    regional_sample = np.concatenate(regional_sample, axis=1)  # (8, 34, 66, 70)

    doy_all    = np.load(doy_path)                     # (1827,)  or similar
    doy_sample = doy_all[window].reshape(8, 5).mean(axis=1)  # (8,) pentad-mean doy

    return regional_sample, doy_sample


def get_climatology_data(date_string,
        clima_dir='/public/home/qinbo/EAAC_V3/data/clima/t2m'):
    """
    Load 40-day climatology window centred on date_string,
    pentad-average → (8, 66, 70).
    """
    input_date = datetime.strptime(date_string, '%Y%m%d')
    lat_slice  = slice(25, 91)
    lon_slice  = slice(70, 141)

    clima_list = []
    for offset in range(-10, 30):
        target = input_date + timedelta(days=offset)
        fpath  = os.path.join(clima_dir, f"t2m_clima_{target.strftime('%m%d')}.npy")
        try:
            arr = np.load(fpath)[lat_slice, lon_slice]   # (66, 71)
        except FileNotFoundError:
            print(f"Warning: {fpath} not found, filling with zeros.")
            arr = np.zeros((66, 71), dtype=np.float32)
        clima_list.append(arr)

    clima_8 = np.stack(clima_list).reshape(8, 5, 66, 71).mean(axis=1)  # (8, 66, 71)
    return clima_8[:, :, :70]                                           # (8, 66, 70)


def get_ecmwf_data_from_date_idx(date_idx):
    """Return (cf, pf, truth) each shaped (6, 66, 70), pentad-averaged."""
    ecmwf_dir = '/public/home/qinbo/EAAC_V3/data/ecmwf_s2s'
    dates     = np.load(f'{ecmwf_dir}/dates.npy')
    idx       = dates.tolist().index(date_idx)

    def _load_pentad(fname):
        arr = np.load(f'{ecmwf_dir}/{fname}')[idx]      # (30, 66, 71)
        return arr.reshape(6, 5, 66, 71).mean(axis=1)[:, :, :70]

    return _load_pentad('cf_forecast.npy'), \
           _load_pentad('pf_mean_forecast.npy'), \
           _load_pentad('ground_truth.npy')


cfg_EAAC_V2 = {
    # I/O shapes
    "in_shape"     : (2, 34, 66, 70),
    "out_shape"    : (1, 34, 66, 70),

    # Architecture
    "C_S"          : 256,    # encoder/decoder hidden channels
    "C_T"          : 768,    # transformer hidden dim
    "N_S"          : 4,      # ConvSC layers in encoder/decoder
    "N_blks"       : 16,     # ST Blocks
    "tc_embed_dim" : 128,    # timecode embedding dim
    "num_heads"    : 12,     # 768 / 12 = 64 per head
    "mlp_ratio"    : 4,
    "drop"         : 0.,
    "drop_path"    : 0.1,

    # Training (used by Lightning module, ignored by EAAC itself)
    "max_epochs"   : 100,
    "lr"           : 1e-4,
    "ckpt_path"    : "/public/home/qinbo/EAAC_V3/lightning_logs/version_2577642/"
                     "checkpoints/exp06-epoch=20-val_loss=0.079675.ckpt",
    "exp_name"     : "exp06",
    "batch_size"   : 8,
    "batch_size_val": 8,
    "devices"      : -1,
    "accumulate_grad_batches": 1,
    "log_dir"      : "/public/home/qinbo/EAAC_V3",
}