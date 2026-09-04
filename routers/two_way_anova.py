from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
import pandas as pd
import scipy.stats as stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

router = APIRouter(prefix="/test", tags=["Parametrik Testler"])

class TwoWayAnovaRequest(BaseModel):
    a: List[str]
    b: List[str]
    y: List[float]

@router.post("/two-way-anova")
def two_way_anova_test(request: TwoWayAnovaRequest):
    # 1. Veri Doğrulama ve DataFrame Oluşturma
    if not (len(request.a) == len(request.b) == len(request.y)):
        raise HTTPException(status_code=400, detail="Faktör A, Faktör B ve Y (Değer) veri sayıları birbirine eşit olmalıdır.")
    
    df = pd.DataFrame({"A": request.a, "B": request.b, "Y": request.y})
    df = df.dropna() # Eksik verileri at
    
    if len(df) < 4:
        raise HTTPException(status_code=400, detail="Analiz için en az 4 geçerli gözlem gereklidir.")

    levels_a = df["A"].nunique()
    levels_b = df["B"].nunique()

    if levels_a < 2 or levels_b < 2:
        raise HTTPException(status_code=400, detail="Her iki faktörün de en az 2'şer seviyesi (grubu) olmalıdır.")

    try:
        # 2. OLS Modeli ve Tip III (Type III) Varyans Analizi
        # Dengesiz tasarımlar için Type III kullanmak istatistiksel standarttır.
        model = ols('Y ~ C(A) + C(B) + C(A):C(B)', data=df).fit()
        aov_table = sm.stats.anova_lm(model, typ=3)
        
        # statsmodels tablosundan verileri güvenle çıkarma fonksiyonu
        def extract_stats(row_name):
            if row_name not in aov_table.index:
                return {"ss": 0.0, "df": 0, "ms": 0.0, "f": 0.0, "p": 1.0, "eta_p": 0.0}
            
            ss = float(aov_table.loc[row_name, 'sum_sq'])
            df_val = int(aov_table.loc[row_name, 'df'])
            ms = ss / df_val if df_val > 0 else 0.0
            
            f_val = float(aov_table.loc[row_name, 'F']) if not np.isnan(aov_table.loc[row_name, 'F']) else 0.0
            p_val = float(aov_table.loc[row_name, 'PR(>F)']) if not np.isnan(aov_table.loc[row_name, 'PR(>F)']) else 1.0
            
            # Kısmi Eta Kare (Partial Eta Squared)
            ss_error = float(aov_table.loc['Residual', 'sum_sq'])
            eta_p = ss / (ss + ss_error) if (ss + ss_error) > 0 else 0.0
            
            return {"ss": ss, "df": df_val, "ms": ms, "f": f_val, "p": p_val, "eta_p": eta_p}

        anova_res = {
            "A": extract_stats("C(A)"),
            "B": extract_stats("C(B)"),
            "AB": extract_stats("C(A):C(B)"),
            "Error": {
                "ss": float(aov_table.loc['Residual', 'sum_sq']),
                "df": int(aov_table.loc['Residual', 'df']),
                "ms": float(aov_table.loc['Residual', 'sum_sq'] / aov_table.loc['Residual', 'df']) if int(aov_table.loc['Residual', 'df']) > 0 else 0.0
            }
        }

        # 3. Model Varsayımları (Shapiro-Wilk ve Levene)
        # Normallik: Modelin artıklarına (residuals) uygulanır.
        residuals = model.resid
        shapiro_w, shapiro_p = stats.shapiro(residuals)
        
        # Levene Varyans Homojenliği
        groups_y = [group["Y"].values for name, group in df.groupby(["A", "B"]) if len(group["Y"].values) > 1]
        if len(groups_y) > 1:
            lev_w, lev_p = stats.levene(*groups_y, center='median')
        else:
            lev_w, lev_p = 0.0, 1.0 # Hücrelerde yeterli varyans yoksa

        # 4. Post-Hoc Testleri (Tukey HSD)
        posthoc_res = {"performed": False, "test_name": "Tukey HSD", "results": []}
        
        # Eğer Faktör A anlamlıysa ve 2'den fazla seviyesi varsa Tukey yap
        if anova_res["A"]["p"] < 0.05 and levels_a > 2:
            posthoc_res["performed"] = True
            tukey_a = pairwise_tukeyhsd(endog=df['Y'], groups=df['A'], alpha=0.05)
            for res in tukey_a.summary().data[1:]:
                posthoc_res["results"].append({
                    "factor": "Faktör A", "group1": str(res[0]), "group2": str(res[1]),
                    "mean_diff": float(res[2]), "p_adj": float(res[3]), 
                    "ci_low": float(res[4]), "ci_up": float(res[5])
                })

        # Eğer Faktör B anlamlıysa ve 2'den fazla seviyesi varsa Tukey yap
        if anova_res["B"]["p"] < 0.05 and levels_b > 2:
            posthoc_res["performed"] = True
            tukey_b = pairwise_tukeyhsd(endog=df['Y'], groups=df['B'], alpha=0.05)
            for res in tukey_b.summary().data[1:]:
                posthoc_res["results"].append({
                    "factor": "Faktör B", "group1": str(res[0]), "group2": str(res[1]),
                    "mean_diff": float(res[2]), "p_adj": float(res[3]), 
                    "ci_low": float(res[4]), "ci_up": float(res[5])
                })

        return {
            "anova": anova_res,
            "assumptions": {
                "shapiro": {"w_stat": float(shapiro_w), "p_value": float(shapiro_p)},
                "levene": {"w_stat": float(lev_w), "p_value": float(lev_p)}
            },
            "posthoc": posthoc_res
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Two-Way ANOVA Hatası: {str(e)}")
