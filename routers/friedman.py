from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import scipy.stats as stats
import numpy as np

router = APIRouter(prefix="/test", tags=["Non-Parametrik Testler"])

class FriedmanRequest(BaseModel):
    conditions: List[List[float]]
    condition_names: List[str]
    p_adj_method: str = "holm"

@router.post("/friedman")
def friedman_test(request: FriedmanRequest):
    try:
        conds = request.conditions
        names = request.condition_names
        k = len(conds)
        
        if k < 3:
            raise HTTPException(status_code=400, detail="Friedman testi için en az 3 koşul gereklidir.")
            
        n = len(conds[0])
        for c in conds:
            if len(c) != n:
                raise HTTPException(status_code=400, detail="Tüm koşulların uzunlukları eşit olmalıdır (Missing veriler atılmalıdır).")

        if n < 3:
            raise HTTPException(status_code=400, detail="Yetersiz denek sayısı.")

        # Friedman Calculation
        stat, p_val = stats.friedmanchisquare(*conds)
        
        # Kendall's W Calculation (Effect Size)
        kendall_w = stat / (n * (k - 1)) if n > 1 and k > 1 else 0

        # Post-hoc pairwise Wilcoxon signed-rank test
        posthoc_res = {"performed": False, "test_name": f"Wilcoxon ({request.p_adj_method.capitalize()})", "results": []}
        
        if p_val < 0.05:
            posthoc_res["performed"] = True
            
            raw_p_values = []
            pairs = []
            
            for i in range(k):
                for j in range(i+1, k):
                    # Eşleştirilmiş olduğu için Wilcoxon kullanıyoruz
                    # Sıfır farkları (ties) ele alırken wilcoxon "pratt" methodu daha güvenlidir
                    w_stat, p_raw = stats.wilcoxon(conds[i], conds[j], method="approx")
                    raw_p_values.append(p_raw)
                    pairs.append((i, j, w_stat))

            # Multiple Comparison Correction (Manuel Holm/Bonferroni)
            m = len(raw_p_values)
            adj_p_values = []
            
            if request.p_adj_method == "bonferroni":
                adj_p_values = [min(1.0, p * m) for p in raw_p_values]
            elif request.p_adj_method == "holm":
                # Holm-Bonferroni
                sorted_indices = np.argsort(raw_p_values)
                adj_p = np.zeros(m)
                for step, idx in enumerate(sorted_indices):
                    adj_p[idx] = min(1.0, raw_p_values[idx] * (m - step))
                # Enforce monotonicity
                for step in range(1, m):
                    idx = sorted_indices[step]
                    prev_idx = sorted_indices[step-1]
                    adj_p[idx] = max(adj_p[prev_idx], adj_p[idx])
                adj_p_values = adj_p.tolist()
            else: # FDR (Benjamini-Hochberg)
                sorted_indices = np.argsort(raw_p_values)
                adj_p = np.zeros(m)
                for step, idx in enumerate(sorted_indices):
                    adj_p[idx] = min(1.0, raw_p_values[idx] * m / (step + 1))
                # Enforce monotonicity reverse
                for step in range(m-2, -1, -1):
                    idx = sorted_indices[step]
                    next_idx = sorted_indices[step+1]
                    adj_p[idx] = min(adj_p[idx], adj_p[next_idx])
                adj_p_values = adj_p.tolist()

            for idx, (i, j, w) in enumerate(pairs):
                posthoc_res["results"].append({
                    "c1": names[i], "c2": names[j], 
                    "stat": float(w), 
                    "p_raw": float(raw_p_values[idx]), "p_adj": float(adj_p_values[idx])
                })

        return {
            "h_stat": float(stat),
            "p_value": float(p_val),
            "kendall_w": float(kendall_w),
            "posthoc": posthoc_res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Friedman Hatası: {str(e)}")
