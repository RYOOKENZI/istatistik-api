from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

router = APIRouter(prefix="/test", tags=["t-Testleri"])

# İstek (Request) Şeması
class TwoSampleTRequest(BaseModel):
    group_1: List[float]
    group_2: List[float]
    alternative: str = "two-sided"
    conf_level: float = 0.95
    var_equal: str = "auto"

@router.post("/independent-t")
def independent_t_test(request: TwoSampleTRequest):
    try:
        arr1 = np.array(request.group_1, dtype=float)
        arr2 = np.array(request.group_2, dtype=float)
        alt = request.alternative
        
        # 1. HATA KONTROLLERİ
        if len(arr1) < 2 or len(arr2) < 2:
            raise HTTPException(status_code=400, detail="Her iki grupta da en az 2'şer gözlem bulunmalıdır.")
            
        n1, n2 = len(arr1), len(arr2)
        mean1, mean2 = np.mean(arr1), np.mean(arr2)
        v1, v2 = np.var(arr1, ddof=1), np.var(arr2, ddof=1)
        
        # 2. LEVENE TESTİ VE VARYANS HOMOJENLİĞİ KARARI
        levene_stat, levene_p = stats.levene(arr1, arr2)
        if request.var_equal == "true":
            equal_var = True
        elif request.var_equal == "false":
            equal_var = False
        else:
            # "auto" seçildiyse Levene testine göre karar ver (p > 0.05 ise varyanslar eşit kabul edilir)
            equal_var = (levene_p > 0.05)

        # 3. SCIPY İLE t-TESTİ HESABI
        res = stats.ttest_ind(arr1, arr2, equal_var=equal_var, alternative=alt)
        t_stat = float(res.statistic)
        p_val = float(res.pvalue)

        # 4. SERBESTLİK DERECESİ (df), STANDART HATA VE COHEN'S d
        if equal_var:
            df = n1 + n2 - 2
            sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / df)
            se_diff = sp * np.sqrt(1/n1 + 1/n2)
            cohen_d = abs(mean1 - mean2) / (sp if sp > 0 else 1e-10)
        else:
            se_diff = np.sqrt(v1/n1 + v2/n2)
            # Welch-Satterthwaite denklemi ile serbestlik derecesi düzeltmesi
            df = (se_diff**4) / (((v1/n1)**2) / (n1 - 1) + ((v2/n2)**2) / (n2 - 1))
            sp = np.sqrt((v1 + v2) / 2)  # Cohen's d için ortalama varyans yaklaşımı
            cohen_d = abs(mean1 - mean2) / (sp if sp > 0 else 1e-10)

        # 5. GÜVEN ARALIĞI (Confidence Interval)
        alpha = 1.0 - request.conf_level
        t_crit = stats.t.ppf(1.0 - alpha/2.0, df)
        diff_mean = float(mean1 - mean2)
        ci_lower = diff_mean - t_crit * se_diff
        ci_upper = diff_mean + t_crit * se_diff

        # 6. TEST GÜCÜ (Statistical Power - Normal Dağılım Yaklaşımı)
        n_eff = (n1 * n2) / (n1 + n2)
        ncp = cohen_d * np.sqrt(n_eff)
        z_crit = stats.norm.ppf(1.0 - alpha/2.0)
        power_val = float(stats.norm.sf(z_crit - ncp) + stats.norm.cdf(-z_crit - ncp))
        power_val = min(1.0, max(0.0, power_val))

        return {
            "test_type": "Student t-Testi" if equal_var else "Welch t-Testi",
            "t_statistic": t_stat,
            "p_value": p_val,
            "df": float(df),
            "cohen_d": float(cohen_d),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "power": power_val,
            "levene_p_value": float(levene_p)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SciPy/Python Hesaplama Hatası: {str(e)}")
