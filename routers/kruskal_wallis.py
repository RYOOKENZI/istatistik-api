from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import scipy.stats as stats

router = APIRouter(prefix="/test", tags=["Non-Parametrik Testler"])

class KruskalRequest(BaseModel):
    groups: List[List[float]]
    group_names: List[str]

@router.post("/kruskal-wallis")
def kruskal_wallis_test(request: KruskalRequest):
    try:
        groups = request.groups
        names = request.group_names
        k = len(groups)
        
        if k < 2:
            raise HTTPException(status_code=400, detail="En az 2 grup gereklidir.")
            
        N = sum([len(g) for g in groups])
        if N < 4:
            raise HTTPException(status_code=400, detail="Yetersiz veri.")

        # Kruskal-Wallis Calculation
        h_stat, p_val = stats.kruskal(*groups)
        
        # Effect Size (Epsilon Squared)
        epsilon_sq = h_stat / (N - 1) if N > 1 else 0

        # Post-hoc pairwise Mann-Whitney U with Bonferroni correction
        posthoc_res = {"performed": False, "test_name": "Mann-Whitney U (Bonferroni)", "results": []}
        
        if p_val < 0.05:
            posthoc_res["performed"] = True
            comparisons = k * (k - 1) / 2
            
            for i in range(k):
                for j in range(i+1, k):
                    stat, p_raw = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                    p_adj = min(1.0, p_raw * comparisons)
                    
                    # Ranks difference (approximation for reporting)
                    diff = (sum(stats.rankdata(groups[i] + groups[j])[:len(groups[i])]) / len(groups[i])) - \
                           (sum(stats.rankdata(groups[i] + groups[j])[len(groups[i]):]) / len(groups[j]))

                    posthoc_res["results"].append({
                        "g1": names[i], "g2": names[j], 
                        "diff": float(diff), 
                        "p_raw": float(p_raw), "p_adj": float(p_adj)
                    })

        return {
            "h_stat": float(h_stat),
            "p_value": float(p_val),
            "epsilon_sq": float(epsilon_sq),
            "posthoc": posthoc_res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Kruskal Hatası: {str(e)}")
