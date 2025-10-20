# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from arch import arch_model


def garch_forecast(
    rets: pd.Series,
    start_date,
    end_date,
    model='GARCH',
    mean='Constant',
    horizon=21,
    lookback=756,
    p=1,
    q=1,
) -> pd.Series:
    """
    Garch Variance Forecast
    """

    r = rets.dropna().copy()
    if not isinstance(r.index, pd.DatetimeIndex):
        r.index = pd.to_datetime(r.index)

    sl = r.index.slice_indexer(pd.Timestamp(start_date), pd.Timestamp(end_date))
    start_pos, end_pos = sl.start, sl.stop - 1
    if start_pos is None or end_pos is None or end_pos < start_pos:
        raise ValueError("Start/end dates not found or empty slice for given index.")

    out_idx = r.index[start_pos:end_pos+1]
    out_vals = np.empty(len(out_idx), dtype=float)

    for j, t in enumerate(range(start_pos, end_pos+1)):
        if lookback is None:
            r_train = r.iloc[:t]
        else:
            r_train = r.iloc[max(0, t - lookback):t]

        if len(r_train) < 100:
            out_vals[j] = np.nan
            continue

        fm = arch_model(r_train, mean=mean, vol=model, p=p, q=q)
        res = fm.fit(disp="off")

        f = res.forecast(horizon=horizon, reindex=False)
        var_path = f.variance.iloc[0].to_numpy()
        cum_var = var_path.sum()
        out_vals[j] = cum_var * (252.0 / horizon)

    return pd.Series(out_vals, index=out_idx, name=f"exp_var_{horizon}d_ann")


def regime_indicator(
    implied_vol: pd.Series, 
    lookback = 756, 
    hi_percentile = 80,
    smooth_window = 5,
    threshold = 0.5,
):
    """
    Regime Indicator based on Implied Volatility
    """
    p_hi = implied_vol.rolling(lookback, min_periods=252).quantile(hi_percentile / 100).shift(1)
    HighVolRisk = (implied_vol > p_hi).astype(float)

    HighVolRisk[p_hi.isna()] = np.nan

    HighVolRisk_smoothed = (
        HighVolRisk.rolling(smooth_window, min_periods=1)
        .mean()
        .gt(threshold)
        .astype(float)
    )

    regime_scale = (1.0 * (HighVolRisk_smoothed == 0)) + (0.0 * (HighVolRisk_smoothed == 1))
    regime_scale[p_hi.isna()] = np.nan

    return regime_scale


