from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import scipy.stats as stats
import numpy as np

router = APIRouter(prefix="/test", tags=["Parametrik Testler"])

class AnovaRequest(BaseModel):
    groups: List[List[float]]
    group_names: List[str]
    method: str = "auto"

@router.post("/one-way-anova")
def one_way_anova(request: AnovaRequest):
    try:
        # Verileri numpy dizisine çevir
        groups = [np.array(g, dtype=float) for g in request.groups]
        names = request.group_names
        k = len(groups)
        
        # Güvenlik Kontrolleri
        if k < 2: 
            raise HTTPException(status_code=400, detail="En az 2 grup gereklidir.")
        for i, g in enumerate(groups):
            if len(g) < 2: 
                raise HTTPException(status_code=400, detail=f"'{names[i]}' grubunda en az 2 veri olmalıdır.")
            if np.var(g, ddof=1) == 0: 
                raise HTTPException(status_code=400, detail=f"'{names[i]}' grubunun varyansı 0. ANOVA yapılamaz.")

        N = sum(len(g) for g in groups)

        # 1. VARSAYIM KONTROLLERİ (Shapiro-Wilk ve Levene)
        shapiro_results = []
        for i, g in enumerate(groups):
            # Shapiro-Wilk testi min 3 veri ister, 2 veri varsa pas geçilir.
            if len(g) >= 3:
                w, p = stats.shapiro(g)
            else:
                w, p = 1.0, 1.0 
            shapiro_results.append({"group": names[i], "w_stat": float(w), "p_value": float(p)})
            
        # Levene (Brown-Forsythe) Varyans Homojenliği
        lev_stat, lev_p = stats.levene(*groups, center='median')
        df_lev_1 = k - 1
        df_lev_2 = N - k

        # Model Seçimi (Otomatikse Levene'e göre karar ver)
        method = request.method
        if method == "auto":
            method = "classic" if lev_p >= 0.05 else "welch"

        # 2. ANOVA HESAPLAMASI
        anova_res = {"df1": float(df_lev_1), "df2": float(df_lev_2)}
        
        if method == "classic":
            f_val, p_val = stats.f_oneway(*groups)
            anova_res.update({
                "method_used": "Klasik Tek Yönlü ANOVA", 
                "f_stat": float(f_val), 
                "p_value": float(p_val)
            })
        else:
            # WELCH ANOVA (Eşit Olmayan Varyanslar İçin)
            n_i = np.array([len(g) for g in groups])
            mean_i = np.array([np.mean(g) for g in groups])
            var_i = np.array([np.var(g, ddof=1) for g in groups])
            
            w_i = n_i / var_i
            sum_w = np.sum(w_i)
            grand_mean_w = np.sum(w_i * mean_i) / sum_w
            
            f_num = np.sum(w_i * (mean_i - grand_mean_w)**2) / (k - 1)
            Lambda = np.sum((1 - w_i/sum_w)**2 / (n_i - 1))
            f_den = 1 + (2 * (k - 2) / (k**2 - 1)) * Lambda
            
            f_val = f_num / f_den
            df1 = k - 1
            df2 = (k**2 - 1) / (3 * Lambda)
            p_val = stats.f.sf(f_val, df1, df2)
            
            anova_res.update({
                "method_used": "Welch ANOVA (Eşit Olmayan Varyans)", 
                "f_stat": float(f_val), 
                "p_value": float(p_val), 
                "df1": float(df1), 
                "df2": float(df2)
            })

        # 3. POST-HOC (Çoklu Karşılaştırmalar)
        posthoc_res = {"performed": False, "test_name": "", "results": []}
        
        # Sadece p < 0.05 ise Post-Hoc yap
        if anova_res["p_value"] < 0.05:
            posthoc_res["performed"] = True
            res_list = []
            
            if method == "classic":
                # TUKEY HSD
                posthoc_res["test_name"] = "Tukey HSD"
                res = stats.tukey_hsd(*groups)
                
                for i in range(k):
                    for j in range(i+1, k):
                        mean_diff = res.statistic[i, j]
                        ci_l = res.confidence_interval.low[i, j]
                        ci_u = res.confidence_interval.high[i, j]
                        p_adj = res.pvalue[i, j]
                        res_list.append({
                            "group1": names[i], "group2": names[j], 
                            "mean_diff": float(mean_diff), 
                            "ci_low": float(ci_l), "ci_up": float(ci_u), 
                            "p_adj": float(p_adj)
                        })
            else:
                # GAMES-HOWELL (Yaklaşımı: Welch T-Test + Bonferroni Correction)
                posthoc_res["test_name"] = "Welch T-Testi (Bonferroni Düzeltmeli)"
                comparisons = k * (k - 1) / 2
                
                for i in range(k):
                    for j in range(i+1, k):
                        t_stat, p_raw = stats.ttest_ind(groups[i], groups[j], equal_var=False)
                        p_adj = min(1.0, p_raw * comparisons) # Bonferroni
                        
                        diff = np.mean(groups[i]) - np.mean(groups[j])
                        se = np.sqrt(np.var(groups[i], ddof=1)/len(groups[i]) + np.var(groups[j], ddof=1)/len(groups[j]))
                        margin = stats.t.ppf(0.975, df_lev_2) * se # Yaklaşık Güven Aralığı
                        
                        res_list.append({
                            "group1": names[i], "group2": names[j], 
                            "mean_diff": float(diff), 
                            "ci_low": float(diff-margin), "ci_up": float(diff+margin), 
                            "p_adj": float(p_adj)
                        })
                        
            posthoc_res["results"] = res_list

        return {
            "anova": anova_res,
            "assumptions": {
                "shapiro": shapiro_results,
                "levene": {
                    "w_stat": float(lev_stat), 
                    "df": f"{df_lev_1}, {df_lev_2}", 
                    "p_value": float(lev_p)
                }
            },
            "posthoc": posthoc_res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python ANOVA Hatası: {str(e)}")
