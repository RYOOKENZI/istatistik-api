# routers/ts_autocorrelation.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# statsmodels kütüphaneleri
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import norm

router = APIRouter(prefix="/test", tags=["Zaman Serisi ACF/PACF"])

# HTML'deki let reqBody = { y: cleanY, nlags: maxLag, alpha: alfa }; yapısına BİREBİR uyumlu model
class AutocorrelationRequest(BaseModel):
    y: List[Optional[float]]
    nlags: int = 40  
    alpha: float = 0.05 

@router.post("/autocorrelation")
def calculate_autocorrelation(req: AutocorrelationRequest):
    try:
        series = pd.Series(req.y).dropna()
        n_obs = len(series)

        if n_obs < 10: 
            raise ValueError("Otokorelasyon hesabı için çok az veri var.")

        max_lags = req.nlags
        if max_lags >= n_obs / 2: 
            max_lags = int(n_obs / 2) - 1
        if max_lags < 1: max_lags = 1

        # 1. Durbin-Watson İstatistiği (HTML rRes.dw bekliyor)
        dw_stat = float(durbin_watson(series))

        # 2. ACF ve PACF Hesaplamaları
        acf_vals, acf_conf = acf(series, nlags=max_lags, alpha=req.alpha, fft=True)
        pacf_vals, pacf_conf = pacf(series, nlags=max_lags, alpha=req.alpha, method='yw')

        # Z değerini (Güven Sınırı Çarpanı) buluyoruz (Örn: 0.05 için 1.96)
        z_val = abs(norm.ppf(req.alpha / 2))

        acf_list = []
        pacf_list = []

        # HTML'in beklediği ACF ve PACF tablosunu döngüyle hazırlıyoruz
        for i in range(max_lags + 1):
            # Standart Hata (SE) hesaplaması (HTML a.se ve p.se bekliyor)
            a_se = float(acf_conf[i, 1] - acf_vals[i]) / z_val if z_val > 0 else 0
            p_se = float(pacf_conf[i, 1] - pacf_vals[i]) / z_val if z_val > 0 else 0
            
            if i == 0:
                a_se, p_se = 0.0, 0.0

            acf_list.append({
                "lag": i,
                "value": float(acf_vals[i]),
                "se": a_se,
                "lower": float(acf_conf[i, 0] - acf_vals[i]),
                "upper": float(acf_conf[i, 1] - acf_vals[i])
            })
            
            pacf_list.append({
                "lag": i,
                "value": float(pacf_vals[i]),
                "se": p_se,
                "lower": float(pacf_conf[i, 0] - pacf_vals[i]),
                "upper": float(pacf_conf[i, 1] - pacf_vals[i])
            })

        # 3. Ljung-Box ve Box-Pierce Testleri (HTML rRes.lb array bekliyor)
        lags_to_test = [l for l in [5, 10, 15, max_lags] if l > 0 and l <= max_lags]
        lags_to_test = list(sorted(set(lags_to_test))) # Tekrarları sil ve sırala

        lb_list = []
        if lags_to_test:
            # boxpierce=True parametresi ile hem Ljung-Box hem de Box-Pierce hesaplanır
            lb_df = acorr_ljungbox(series, lags=lags_to_test, return_df=True, boxpierce=True)
            for lag in lags_to_test:
                lb_list.append({
                    "lag": int(lag),
                    "lb_stat": float(lb_df.loc[lag, 'lb_stat']),
                    "lb_p": float(lb_df.loc[lag, 'lb_pvalue']),
                    "bp_stat": float(lb_df.loc[lag, 'bp_stat']),
                    "bp_p": float(lb_df.loc[lag, 'bp_pvalue'])
                })

        # Bütün veriyi HTML'in beklediği yapıda JSON olarak döndürüyoruz
        return {
            "acf": acf_list,
            "pacf": pacf_list,
            "lb": lb_list,
            "dw": dw_stat
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Otokorelasyon Analizi Hatası: {str(e)}")
