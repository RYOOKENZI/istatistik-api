# routers/ts_autocorrelation.py
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

        if n_obs < 10: 
            raise ValueError("Otokorelasyon hesabı için çok az veri var.")

        max_lags = min(req.lags, int(n_obs / 2) - 1)
        if max_lags < 1: max_lags = 1

        # statsmodels ile ACF ve PACF hesaplamaları
        acf_vals, acf_conf, qstats, q_pvals = acf(series, nlags=max_lags, alpha=0.05, qstat=True, fft=True)
        pacf_vals, pacf_conf = pacf(series, nlags=max_lags, alpha=0.05, method='yw')

        # Toplu Ljung-Box Testi
        lb_df = acorr_ljungbox(series, lags=[max_lags], return_df=True)
        lb_stat = float(lb_df['lb_stat'].iloc[0])
        lb_pval = float(lb_df['lb_pvalue'].iloc[0])

        acf_list = []
        pacf_list = []

        for i in range(max_lags + 1):
            # İlk gecikme (lag=0) için istatistik olmaz, 0 ve 1 veriyoruz.
            q_val = float(qstats[i-1]) if i > 0 else 0.0
            p_val = float(q_pvals[i-1]) if i > 0 else 1.0

            # Matematiksel boşlukları (NaN) sıfıra çevirme koruması
            if np.isnan(q_val): q_val = 0.0
            if np.isnan(p_val): p_val = 1.0

            # JavaScript çökmesin diye tüm olası isim varyasyonlarını "zırh" olarak ekliyoruz
            acf_list.append({
                "lag": i,
                "value": float(acf_vals[i]),
                "lower": float(acf_conf[i, 0] - acf_vals[i]),
                "upper": float(acf_conf[i, 1] - acf_vals[i]),
                "q": q_val,             # HTML tablosu 'q' arıyorsa
                "q_stat": q_val,        # HTML 'q_stat' arıyorsa
                "p": p_val,             # HTML 'p' arıyorsa
                "p_value": p_val        # HTML 'p_value' arıyorsa
            })
            
            pacf_list.append({
                "lag": i,
                "value": float(pacf_vals[i]),
                "lower": float(pacf_conf[i, 0] - pacf_vals[i]),
                "upper": float(pacf_conf[i, 1] - pacf_vals[i])
            })

        return {
            "acf": acf_list,
            "pacf": pacf_list,
            "ljung_box": {
                "stat": lb_stat,
                "q": lb_stat,
                "p_value": lb_pval, 
                "p": lb_pval,
                "lag": max_lags
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otokorelasyon Analizi Hatası: {str(e)}")
