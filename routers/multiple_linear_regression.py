from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

router = APIRouter(prefix="/test", tags=["Regresyon Modelleri"])

class MultipleLinearRegressionRequest(BaseModel):
    x: Dict[str, List[float]] # Birden fazla X kolonu
    y: List[float]
    alpha: float = 0.05
    conf_level: float = 0.95

@router.post("/multiple-linear-regression")
def multiple_linear_regression_test(request: MultipleLinearRegressionRequest):
    try:
        # DataFrame Oluşturma
        df_dict = {'Y_Bagimli': request.y}
        x_names = list(request.x.keys())
        for xn in x_names:
            df_dict[xn] = request.x[xn]
            
        df = pd.DataFrame(df_dict)
        df = df.dropna()
        n = len(df)
        k = len(x_names)
        
        if n < k + 2:
            raise HTTPException(status_code=400, detail=f"Gözlem sayısı (n={n}), bağımsız değişken sayısından (k={k}) yetersizdir. Serbestlik derecesi sağlanamıyor.")
            
        # OLS Model Fitting
        X_df = df[x_names]
        X = sm.add_constant(X_df) # Sabit (Intercept) ekle
        model = sm.OLS(df['Y_Bagimli'], X).fit()
        
        # Temel İstatistikler
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        f_stat = float(model.fvalue) if not np.isnan(model.fvalue) else 0.0
        f_p = float(model.f_pvalue) if not np.isnan(model.f_pvalue) else 1.0

        # Kareler Toplamı (Sum of Squares)
        sst = float(model.centered_tss)
        sse = float(model.ssr)
        ssr = float(model.ess)
        mse = float(model.mse_resid)
        
        # Güven Aralıkları
        conf_int = model.conf_int(alpha=1 - request.conf_level)
        
        # Standartlaştırılmış Beta hesaplama (Y = b * Sx/Sy)
        # Sadece X değişkenleri için hesaplanır (Sabit hariç)
        sy = df['Y_Bagimli'].std()
        std_betas = []
        for xn in x_names:
            sx = df[xn].std()
            b = model.params[xn]
            std_betas.append(float(b * (sx / sy) if sy != 0 else 0))

        # VIF (Variance Inflation Factor) Hesaplama
        vif_data = []
        for i in range(1, X.shape[1]): # Sabiti (index 0) atla
            vif_val = variance_inflation_factor(X.values, i)
            vif_data.append(float(vif_val))
            
        tolerance_data = [1/v if v != 0 else 0 for v in vif_data]

        # Katsayılar Listesi
        coefs = []
        # Sabit (Intercept) - index 0
        coefs.append({
            "name": "Intercept",
            "estimate": float(model.params.iloc[0]),
            "std_error": float(model.bse.iloc[0]),
            "std_beta": 0.0,
            "t": float(model.tvalues.iloc[0]),
            "p_value": float(model.pvalues.iloc[0]),
            "ci_lower": float(conf_int.iloc[0, 0]),
            "ci_upper": float(conf_int.iloc[0, 1])
        })
        # X Değişkenleri
        for idx, xn in enumerate(x_names):
            coefs.append({
                "name": xn,
                "estimate": float(model.params[xn]),
                "std_error": float(model.bse[xn]),
                "std_beta": float(std_betas[idx]),
                "t": float(model.tvalues[xn]),
                "p_value": float(model.pvalues[xn]),
                "ci_lower": float(conf_int.loc[xn, 0]),
                "ci_upper": float(conf_int.loc[xn, 1])
            })

        # Etkili Gözlemler ve Artıklar (Diagnostics)
        influence = model.get_influence()
        resid = model.resid.tolist()
        std_resid = influence.resid_studentized_internal.tolist()
        cooks_d = influence.cooks_distance[0].tolist()
        predictions = model.fittedvalues.tolist()

        # Varsayım Kontrolleri
        # 1. Shapiro-Wilk (Normality of residuals)
        sw_stat, sw_p = stats.shapiro(resid) if n >= 3 else (0.0, 1.0)
        
        # 2. Breusch-Pagan (Homoscedasticity)
        bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)

        return {
            "method": "Multiple Linear Regression",
            "n": n,
            "k": k,
            "r_squared": r2,
            "adjusted_r_squared": adj_r2,
            "f_statistic": f_stat,
            "f_p_value": f_p,
            "anova": {
                "sst": sst, "ssr": ssr, "sse": sse,
                "df_model": k, "df_resid": n - k - 1,
                "msr": ssr / k if k > 0 else 0, "mse": mse
            },
            "coefficients": coefs,
            "vif": vif_data,
            "tolerance": tolerance_data,
            "residuals": resid,
            "predictions": predictions,
            "diagnostics": {
                "std_resid": std_resid,
                "cooks_d": cooks_d,
                "shapiro_w": float(sw_stat),
                "shapiro_p": float(sw_p),
                "bp_lm": float(bp_stat),
                "bp_p": float(bp_p)
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Çoklu Regresyon Hatası: {str(e)}")
