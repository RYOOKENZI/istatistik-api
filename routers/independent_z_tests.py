from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats
import math

router = APIRouter(prefix="/test", tags=["Z-Testleri"])

class TwoSampleZRequest(BaseModel):
    group_1: List[float]
    group_2: List[float]
    alternative: str = "two-sided"
    conf_level: float = 0.95
    d_null: float = 0.0
    pop_sd1: Optional[float] = None
    pop_sd2: Optional[float] = None

@router.post("/independent-z")
def independent_z_test(request: TwoSampleZRequest):
    try:
        arr1 = np.array(request.group_1, dtype=float)
        arr2 = np.array(request.group_2, dtype=float)
        alt = request.alternative
        d0 = request.d_null
        
        n1, n2 = len(arr1), len(arr2)
        if n1 < 2 or n2 < 2:
            raise HTTPException(status_code=400, detail="Her grupta en az 2 veri olmalıdır.")
            
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        
        # Eğer evren varyansları verilmişse onları kullan, verilmediyse örneklem varyanslarını kullan (Büyük Örneklem Yaklaşımı)
        var1 = request.pop_sd1**2 if request.pop_sd1 else np.var(arr1, ddof=1)
        var2 = request.pop_sd2**2 if request.pop_sd2 else np.var(arr2, ddof=1)
        
        se_diff = math.sqrt(var1/n1 + var2/n2)
        if se_diff == 0:
            raise HTTPException(status_code=400, detail="Grupların varyansı sıfır. Test yapılamıyor.")
            
        z_stat = (mean1 - mean2 - d0) / se_diff
        
        # P-değeri hesabı (SciPy ile)
        if alt == "two-sided":
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alt == "greater":
            p_val = 1 - stats.norm.cdf(z_stat)
        else: # less
            p_val = stats.norm.cdf(z_stat)
            
        # Güven Aralığı
        alpha = 1.0 - request.conf_level
        z_crit = stats.norm.ppf(1.0 - alpha/2.0)
        ci_lower = float((mean1 - mean2) - z_crit * se_diff)
        ci_upper = float((mean1 - mean2) + z_crit * se_diff)
        
        # Etki Büyüklüğü (Ortalama Varyans üzerinden Cohen's d)
        sp_cohen = math.sqrt((var1 + var2) / 2)
        cohen_d = abs(mean1 - mean2) / sp_cohen
        
        # Test Gücü (Power - Normal Dağılım)
        n_eff = (n1 * n2) / (n1 + n2)
        ncp = cohen_d * math.sqrt(n_eff)
        power_val = float(stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
        power_val = min(1.0, max(0.0, power_val))

        return {
            "z_statistic": float(z_stat),
            "p_value": float(p_val),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "cohen_d": float(cohen_d),
            "power": power_val
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Z-Testi Hatası: {str(e)}")
