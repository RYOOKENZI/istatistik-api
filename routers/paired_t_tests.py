from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

router = APIRouter(prefix="/test", tags=["t-Testleri"])

class PairedTRequest(BaseModel):
    group_1: List[float]
    group_2: List[float]
    alternative: str = "two-sided"
    conf_level: float = 0.95
    d_null: float = 0.0

@router.post("/paired-t")
def paired_t_test(request: PairedTRequest):
    try:
        arr1 = np.array(request.group_1, dtype=float)
        arr2 = np.array(request.group_2, dtype=float)
        alt = request.alternative
        d0 = request.d_null
        
        if len(arr1) != len(arr2):
            raise HTTPException(status_code=400, detail="Bağımlı t-testinde grupların gözlem sayıları eşit olmalıdır.")
        if len(arr1) < 2:
            raise HTTPException(status_code=400, detail="En az 2 eşleştirilmiş gözlem çifti gereklidir.")
            
        # 1. Fark Puanları (d_i = X_2 - X_1)
        diffs = arr2 - arr1 - d0
        n = len(diffs)
        df = n - 1
        mean_diff = np.mean(diffs)
        sd_diff = np.std(diffs, ddof=1)
        se_diff = sd_diff / np.sqrt(n)
        
        # 2. SciPy t-Testi
        res = stats.ttest_rel(arr2, arr1 + d0, alternative=alt)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)
        
        # 3. Cohen's dz Etki Büyüklüğü
        cohen_dz = abs(mean_diff) / (sd_diff if sd_diff > 0 else 1e-10)
        
        # 4. Güven Aralığı
        alpha = 1.0 - request.conf_level
        t_crit = stats.t.ppf(1.0 - alpha/2.0, df)
        ci_lower = float(mean_diff - t_crit * se_diff)
        ci_upper = float(mean_diff + t_crit * se_diff)
        
        # 5. Test Gücü (Power)
        ncp = cohen_dz * np.sqrt(n)
        z_crit = stats.norm.ppf(1.0 - alpha/2.0)
        power_val = float(stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
        power_val = min(1.0, max(0.0, power_val))

        return {
            "t_statistic": t_stat,
            "p_value": p_val,
            "df": float(df),
            "cohen_dz": float(cohen_dz),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "power": power_val,
            "mean_diff": float(mean_diff)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SciPy Hesaplama Hatası: {str(e)}")
