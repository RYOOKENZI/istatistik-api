from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan

router = APIRouter(prefix="/test", tags=["Regresyon Modelleri"])

class SimpleLinearRegressionRequest(BaseModel):
    x: List[float]
    y: List[float]
    alpha: float = 0.05
    conf_level: float = 0.95

@router.post("/simple-linear-regression")
def simple_linear_regression_test(request: SimpleLinearRegressionRequest):
    try:
        df = pd.DataFrame({'X': request.x, 'Y': request.y})
        df = df.dropna()
        n = len(df)
        
        if n < 3:
            raise HTTPException(status_code=400, detail="Analiz için en az 3 geçerli gözlem gereklidir.")
            
        if df['X'].var() == 0:
            raise HTTPException(status_code=400, detail="Bağımsız değişkenin (X) varyansı sıfırdır. Regresyon hesaplanamaz.")

        # OLS Model Fitting
        X = sm.add_constant(df['X'])
        model = sm.OLS(df['Y'], X).fit()
        
        # Temel İstatistikler
        b0 = float(model.params.iloc[0])
        b1 = float(model.params.iloc[1])
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        f_stat = float(model.fvalue)
        f_p = float(model.f_pvalue)
        
        # Pearson R for summary
        r_val = np.sign(b1) * np.sqrt(r2)

        # Kareler Toplamı (Sum of Squares)
        sst = float(model.centered_tss)
        sse = float(model.ssr)
        ssr = float(model.ess)
        mse = float(model.mse_resid)
        
        # Güven Aralıkları (Coefficients CI)
        conf_int = model.conf_int(alpha=1 - request.conf_level)
        
        # Etkili Gözlemler ve Artıklar (Diagnostics)
        influence = model.get_influence()
        resid = model.resid.tolist()
        std_resid = influence.resid_studentized_internal.tolist() # Standardized/Studentized
        cooks_d = influence.cooks_distance[0].tolist()
        predictions = model.fittedvalues.tolist()

        # Varsayım Kontrolleri (Assumptions)
        # 1. Shapiro-Wilk (Normality of residuals)
        sw_stat, sw_p = stats.shapiro(resid) if n >= 3 else (0.0, 1.0)
        
        # 2. Breusch-Pagan (Homoscedasticity)
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)

        return {
            "method": "Simple Linear Regression",
            "n": n,
            "intercept": b0,
            "slope": b1,
            "r": float(r_val),
            "r_squared": r2,
            "adjusted_r_squared": adj_r2,
            "sse": sse,
            "ssr": ssr,
            "sst": sst,
            "mse": mse,
            "f_statistic": f_stat,
            "f_p_value": f_p,
            "coefficients": {
                "intercept": {
                    "estimate": b0,
                    "std_error": float(model.bse.iloc[0]),
                    "t": float(model.tvalues.iloc[0]),
                    "p_value": float(model.pvalues.iloc[0]),
                    "ci_lower": float(conf_int.iloc[0, 0]),
                    "ci_upper": float(conf_int.iloc[0, 1])
                },
                "slope": {
                    "estimate": b1,
                    "std_error": float(model.bse.iloc[1]),
                    "t": float(model.tvalues.iloc[1]),
                    "p_value": float(model.pvalues.iloc[1]),
                    "ci_lower": float(conf_int.iloc[1, 0]),
                    "ci_upper": float(conf_int.iloc[1, 1])
                }
            },
            "residuals": {
                "resid": resid,
                "std_resid": std_resid,
                "cooks_d": cooks_d
            },
            "predictions": predictions,
            "diagnostics": {
                "shapiro_w": float(sw_stat),
                "shapiro_p": float(sw_p),
                "bp_lm": float(bp_stat),
                "bp_p": float(bp_p)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python OLS Hatası: {str(e)}")
