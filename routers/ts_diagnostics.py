# routers/ts_diagnostics.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
import scipy.stats as stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

router = APIRouter(prefix="/test", tags=["Zaman Serisi Tanıları"])

class DiagnosticsRequest(BaseModel):
    residuals: List[float]
    lags: List[int] = [5, 10, 15]

@router.post("/model-diagnostics")
def calculate_diagnostics(req: DiagnosticsRequest):
    try:
        resid = np.array(req.residuals)
        resid = resid[~np.isnan(resid)] # Null değerleri at
        n = len(resid)

        if n < 5:
            raise ValueError("Tanı testleri için yeterli hata (residual) verisi yok.")

        # 1. TEMEL İSTATİSTİKLER (Ortalama, Çarpıklık, Basıklık)
        mean = float(np.mean(resid))
        std = float(np.std(resid, ddof=1))
        skew = float(stats.skew(resid))
        kurt = float(stats.kurtosis(resid)) # Fisher's kurtosis (normal is 0)

        # 2. LJUNG-BOX (Otokorelasyon Testi)
        # Verilen laglar için test yap (Eğer lag sayısı veri uzunluğundan büyükse sınırla)
        valid_lags = [l for l in req.lags if l < n/2]
        if not valid_lags: valid_lags = [min(5, max(1, int(n/3)))]
        
        lb_res = acorr_ljungbox(resid, lags=valid_lags, return_df=True)
        lb_list = []
        for lag in valid_lags:
            if lag in lb_res.index:
                lb_list.append({"lag": lag, "q": float(lb_res.loc[lag, 'lb_stat']), "p": float(lb_res.loc[lag, 'lb_pvalue'])})

        # 3. ARCH-LM TESTİ (Değişen Varyans)
        arch_list = []
        for lag in valid_lags:
            try:
                # het_arch testi: stat, pval, fstat, fpval döndürür
                arch_stat = het_arch(resid, nlags=lag)
                arch_list.append({"lag": lag, "lm": float(arch_stat[0]), "p": float(arch_stat[1])})
            except:
                arch_list.append({"lag": lag, "lm": 0.0, "p": 1.0})

        # 4. JARQUE-BERA (Normallik Testi)
        jb_stat, jb_pval = stats.jarque_bera(resid)

        return {
            "stats": {"mean": mean, "std": std, "skew": skew, "kurt": kurt},
            "lb": lb_list,
            "arch": arch_list,
            "norm": {"jb": float(jb_stat), "p": float(jb_pval)}
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tanı (Diagnostics) Hatası: {str(e)}")
