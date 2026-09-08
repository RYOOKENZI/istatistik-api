# routers/ts_volatility.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

# arch kütüphanesinin GARCH motoru
from arch import arch_model
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

router = APIRouter(prefix="/test", tags=["Zaman Serisi Volatilite"])

class VolatilityRequest(BaseModel):
    y: List[float] # Burada direkt Getiri (Return) verisi gelmeli
    model: str = 'GARCH' # ARCH, GARCH, EGARCH, GJR-GARCH vb.
    dist: str = 'normal' # normal, studentst, skewstudent
    p: int = 1
    q: int = 1

@router.post("/volatility")
def calculate_volatility(req: VolatilityRequest):
    try:
        # Getiri verisini al (Null değerleri at, log-return vs zaten önyüzde yapıldı)
        y_clean = [val for val in req.y if val is not None and not np.isnan(val)]
        series = pd.Series(y_clean)

        if len(series) < 50:
            raise ValueError("GARCH / Volatilite modelleri için en az 50 gözlem (getiri verisi) gereklidir.")

        # 1. ARCH-LM TESTİ (Model Öncesi Verideki ARCH Etkisi)
        arch_stat = het_arch(series - series.mean(), nlags=5)
        pre_arch_p = float(arch_stat[1]) # p-değeri

        # 2. MODEL TANIMLAMASI
        vol_type = 'GARCH'
        if req.model == 'EGARCH': vol_type = 'EGARCH'
        elif req.model == 'GJR-GARCH': vol_type = 'GARCH' # GJR, GARCH içinde 'o' parametresiyle çalışır
        
        # Arch kütüphanesi GJR-GARCH için "o" (asimetri) parametresi ister.
        o_param = 1 if req.model == 'GJR-GARCH' else 0

        # Modeli Kur (Mean='Constant' yani basit getiri ortalaması etrafında)
        am = arch_model(series, mean='Constant', vol=vol_type, p=req.p, o=o_param, q=req.q, dist=req.dist)
        
        # Modeli Eğit (Yakınsama için uyarıları kapatıyoruz)
        res = am.fit(disp="off")

        # 3. KATSAYILAR (Parametreler)
        coef_list = []
        for name, value in res.params.items():
            p_val = res.pvalues.get(name, 1.0)
            t_val = res.tvalues.get(name, 0.0)
            se_val = res.std_err.get(name, 0.0)
            coef_list.append({
                "name": name, "value": float(value), "se": float(se_val), "t": float(t_val), "p": float(p_val)
            })

        # 4. KOŞULLU VOLATİLİTE VE ARTIKLAR
        cond_vol = res.conditional_volatility.tolist()
        std_resid = (res.resid / res.conditional_volatility).dropna()

        # 5. MODEL TANILARI (Standartlaştırılmış Kareli Artıklarda Ljung-Box ve ARCH-LM)
        sq_std_resid = std_resid ** 2
        lb_res = acorr_ljungbox(sq_std_resid, lags=[5], return_df=True)
        diag_lb_p = float(lb_res['lb_pvalue'].iloc[0])

        diag_arch = het_arch(std_resid, nlags=5)
        diag_arch_p = float(diag_arch[1])

        # Kalıcılık (Persistence) - Alfa + Beta
        persistence = float(res.params.get('alpha[1]', 0) + res.params.get('beta[1]', 0))
        if req.model == 'GJR-GARCH':
            persistence += float(res.params.get('gamma[1]', 0)) / 2 # GJR için ortalama kalıcılık formülü

        return {
            "model_name": f"{req.model}({req.p},{req.q}) [{req.dist.capitalize()}]",
            "ll": float(res.loglikelihood),
            "aic": float(res.aic),
            "bic": float(res.bic),
            "persist": persistence,
            "arch_lm_p": pre_arch_p,
            "coefs": coef_list,
            "volatility": cond_vol,
            "diag_lb_p": diag_lb_p,
            "diag_arch_p": diag_arch_p,
            "diag_jb_p": 0.05 # Geliştirilebilir
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Volatilite Analizi Hatası: {str(e)}")
