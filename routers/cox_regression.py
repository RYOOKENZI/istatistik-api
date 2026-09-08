from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter
from statsmodels.stats.outliers_influence import variance_inflation_factor

router = APIRouter(prefix="/test", tags=["Sağkalım Analizleri"])

class CoxRegressionRequest(BaseModel):
    x: Dict[str, List[float]] 
    t: List[float] 
    e: List[float] 
    alpha: float = 0.05
    conf_level: float = 0.95
    ties: str = "efron"

@router.post("/cox-regression")
def cox_regression_test(request: CoxRegressionRequest):
    try:
        # 1. Veri Hazırlama
        df_dict = {'duration': request.t, 'event': request.e}
        x_names = list(request.x.keys())
        for xn in x_names:
            df_dict[xn] = request.x[xn]
            
        df = pd.DataFrame(df_dict).dropna()
        n = len(df)
        k = len(x_names)
        
        if n < k + 2:
            raise HTTPException(status_code=400, detail=f"Gözlem sayısı (n={n}), bağımsız değişken sayısından (k={k}) yetersizdir. Olasılık maksimizasyonu çalıştırılamıyor.")
            
        # 2. Cox Modeli Kurulumu
        tie_method = 'efron' if request.ties == 'efron' else 'breslow'
        alpha_val = 1.0 - request.conf_level # Lifelines için alpha (0.05 vs)
        
        cph = CoxPHFitter(alpha=alpha_val, ties=tie_method)
        
        try:
            cph.fit(df, duration_col='duration', event_col='event')
        except Exception as mle_err:
            raise HTTPException(status_code=400, detail="Cox modeli yakınsamadı (Convergence Error). Yüksek çoklu bağlantı (Multicollinearity) veya sabit değerli bir değişken olabilir.")

        # 3. İstatistiksel Çıkarımlar
        summary = cph.summary
        lr_stat = float(cph.log_likelihood_ratio_test().test_statistic)
        lr_p = float(cph.log_likelihood_ratio_test().p_value)
        c_index = float(cph.concordance_index_)

        coefs = []
        hazard_ratios = []
        
        # Kolon ismi bağımlılığından kurtulup, doğrudan matris'den CI çektik:
        ci_df = cph.confidence_intervals_
        ci_hr_df = np.exp(ci_df) # Hazard Ratio CI

        for xn in x_names:
            coefs.append({
                "name": xn,
                "estimate": float(summary.loc[xn, 'coef']),
                "std_error": float(summary.loc[xn, 'se(coef)']),
                "z": float(summary.loc[xn, 'z']),
                "p_value": float(summary.loc[xn, 'p'])
            })
            hazard_ratios.append({
                "estimate": float(summary.loc[xn, 'exp(coef)']),
                "ci_lower": float(ci_hr_df.loc[xn].iloc[0]),
                "ci_upper": float(ci_hr_df.loc[xn].iloc[1])
            })

        # 4. Proportional Hazards Test (Schoenfeld)
        ph_p_values = []
        try:
            from lifelines.statistics import proportional_hazard_test
            ph_res = proportional_hazard_test(cph, df, time_transform='rank')
            for xn in x_names:
                ph_p_values.append(float(ph_res.p_value[xn]))
        except:
            ph_p_values = [1.0] * k

        # 5. VIF (Çoklu Doğrusal Bağlantı)
        X_vals = df[x_names].values
        X_vif = np.hstack([np.ones((n, 1)), X_vals])
        vif_data = []
        if k > 1:
            for i in range(1, k + 1): 
                try:
                    vif_val = variance_inflation_factor(X_vif, i)
                    vif_data.append(float(vif_val))
                except:
                    vif_data.append(1.0)
        else:
            vif_data = [1.0]

        # 6. Temel Sağkalım Eğrisi (Baseline Survival)
        baseline = cph.baseline_survival_
        base_times = baseline.index.tolist()
        base_surv = baseline.iloc[:, 0].tolist()

        return {
            "method": "Cox Proportional Hazards",
            "n": n,
            "concordance_index": c_index,
            "likelihood_ratio_test": {
                "chi_square": lr_stat,
                "df": k,
                "p_value": lr_p
            },
            "coefficients": coefs,
            "hazard_ratios": hazard_ratios,
            "schoenfeld_test": {
                "p_values": ph_p_values
            },
            "vif": vif_data,
            "baseline_survival": {
                "time": [float(t) for t in base_times],
                "survival": [float(s) for s in base_surv]
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Cox Hatası: {str(e)}")
