import numpy as np
import pandas as pd


def normalize_fluorescence(F, B,
                           method="dff",      # "dff" or "snr"
                           baseline="auto",   # float, or "auto"
                           roll_seconds=60,   # for auto baseline
                           fps=None,          # needed if baseline="auto" and roll_seconds given
                           low_pct=20,        # rolling percentile for baseline
                           bleach=None,       # None, "exp", or "lin"
                           eps=1e-9):
    """
    F: 1D array of fluorescence (cell ROI)
    B: 1D array of background (bg ROI)
    area_*: ROI pixel counts to scale background
    """
    F = np.asarray(F, float)
    B = np.asarray(B, float)
    assert F.shape == B.shape, "F and B must have same length"

    # per-frame background subtraction
    Fcorr = F - B

    # optional bleach correction
    if bleach is not None:
        t = np.arange(len(Fcorr), dtype=float)
        if bleach == "lin":
            # linear trend
            A = np.vstack([t, np.ones_like(t)]).T
            m, c = np.linalg.lstsq(A, Fcorr, rcond=None)[0]
            trend = m*t + c
            trend = np.clip(trend, eps, None)
            Fcorr = Fcorr / trend * np.median(trend)
        elif bleach == "exp":
            # log-linear fit for exponential bleaching: F ~ a*exp(b t)
            y = np.log(np.clip(Fcorr - np.min(Fcorr) + eps, eps, None))
            A = np.vstack([t, np.ones_like(t)]).T
            b, loga = np.linalg.lstsq(A, y, rcond=None)[0]
            trend = np.exp(loga) * np.exp(b*t)
            trend = np.clip(trend, eps, None)
            Fcorr = Fcorr / trend * np.median(trend)

    # baseline
    if isinstance(baseline, (int, float)):
        F0 = float(baseline)
    else:
        if fps is None:
            # fallback: global low percentile if fps unknown
            F0 = np.percentile(Fcorr, low_pct)
        else:
            win = max(3, int(round(roll_seconds * fps)))
            s = pd.Series(Fcorr)
            F0_series = s.rolling(win, center=True, min_periods=1)\
                         .apply(lambda v: np.percentile(v, low_pct), raw=True)
            # choose a single F0 (median of rolling baseline) for classic ΔF/F0
            F0 = float(np.median(F0_series.values))

    F0 = max(F0, eps)

    if method.lower() == "dff":
        out = (Fcorr - F0) / F0
    elif method.lower() == "snr":
        out = Fcorr / np.maximum(B, eps)
    else:
        raise ValueError("method must be 'dff' or 'snr'")

    return out, Fcorr, F0


def smooth_trace(trace, window=5, method="linear"):
    trace = np.asarray(trace, float)

    # replace zeros with NaN
    trace[trace == 0] = np.nan

    # interpolate missing values
    s = pd.Series(trace)
    trace_interp = s.interpolate(method=method, limit_direction="both").to_numpy()

    # apply moving average smoothing
    smoothed = pd.Series(trace_interp).rolling(window, center=True, min_periods=1).mean().to_numpy()

    return smoothed, trace_interp


def post_process(F, B):
    smoothed, filled = smooth_trace(F, window=3)
    dff, Fcorr, F0 = normalize_fluorescence(
        smoothed, B,
        area_cell=1, area_bg=1,
        method="dff",
        baseline="auto", roll_seconds=60, fps=5,
        low_pct=20,
        bleach=None   # try None, "lin", or "exp"
    )
    return Fcorr
