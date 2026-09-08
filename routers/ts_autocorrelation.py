from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

router = APIRouter(prefix="/test", tags=["Zaman Serisi ACF/PACF"])

class AutocorrelationRequest(BaseModel):
    y: List[Optional[float]]
    lags: int = 40  

@router.post("/autocorrelation")
def calculate_autocorrelation(req: AutocorrelationRequest):
    try:
        series = pd.Series(req.y).dropna()
        n_obs = len(series)

        if n_obs < 10: raise ValueError("Otokorelasyon hesabı için çok az veri var.")

        max_lags = min(req.lags, int(n_obs / 2) - 1)
        if max_lags < 1: max_lags = 1

        acf_vals, acf_conf, qstats, q_pvals = acf(series, nlags=max_lags, alpha=0.05, qstat=True, fft=True)
        pacf_vals, pacf_conf = pacf(series, nlags=max_lags, alpha=0.05, method='yw')

        lb_df = acorr_ljungbox(series, lags=[max_lags], return_df=True)
        lb_stat = float(lb_df['lb_stat'].iloc[0])
        lb_pval = float(lb_df['lb_pvalue'].iloc[0])

        acf_lower = (acf_conf[:, 0] - acf_vals).tolist()
        acf_upper = (acf_conf[:, 1] - acf_vals).tolist()
        pacf_lower = (pacf_conf[:, 0] - pacf_vals).tolist()
        pacf_upper = (pacf_conf[:, 1] - pacf_vals).tolist()

        return {
            "lags": list(range(max_lags + 1)),
            "acf": {"values": acf_vals.tolist(), "lower": acf_lower, "upper": acf_upper},
            "pacf": {"values": pacf_vals.tolist(), "lower": pacf_lower, "upper": pacf_upper},
            "ljung_box": {"stat": lb_stat, "p_value": lb_pval, "lag": max_lags}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otokorelasyon Analizi Hatası: {str(e)}")
