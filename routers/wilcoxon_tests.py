from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

# Router tanımı (Swagger'da "Wilcoxon Testleri" başlığı altında toplanır)
router = APIRouter(prefix="/test", tags=["Wilcoxon Testleri"])

# ---------------------------------------------------------
# 1. TEK ÖRNEKLEM (ONE-SAMPLE) WILCOXON TESTİ
# ---------------------------------------------------------
class OneSampleWilcoxonRequest(BaseModel):
    test_value: float
    alternative: str = "two-sided" 
    conf_level: float = 0.95
    data: List[float]

@router.post("/one-sample-wilcoxon")
def one_sample_wilcoxon_test(request: OneSampleWilcoxonRequest):
    try:
        arr = np.array(request.data)
        m0 = request.test_value
        alt = request.alternative
        
        diffs = arr - m0
        valid_diffs = diffs[diffs != 0]
        n_valid = len(valid_diffs)
        
        if n_valid < 1:
            raise HTTPException(status_code=400, detail="Test medyanına eşit olmayan en az 1 gözlem girmelisiniz.")
        
        abs_diffs = np.abs(valid_diffs)
        ranks = stats.rankdata(abs_diffs, method='average') 
        
        w_plus = np.sum(ranks[valid_diffs > 0])
        w_minus = np.sum(ranks[valid_diffs < 0])
        w_stat = min(w_plus, w_minus)
        
        try:
            res = stats.wilcoxon(valid_diffs, alternative=alt, correction=True, exact=False)
            scipy_p_val = res.pvalue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SciPy Hesaplama Hatası: {str(e)}")

        mu_w = n_valid * (n_valid + 1) / 4.0
        unique_vals, counts = np.unique(abs_diffs, return_counts=True)
        tie_sum = np.sum(counts**3 - counts)
        var_w = (n_valid * (n_valid + 1) * (2 * n_valid + 1)) / 24.0 - tie_sum / 48.0
        sigma_w = np.sqrt(var_w) if var_w > 0 else 0
        
        z_stat = 0.0
        if sigma_w > 0:
            diff_w = w_plus - mu_w
            correction = 0.5 * np.sign(diff_w)
            if abs(diff_w) < 0.5:
                correction = diff_w
            z_stat = (diff_w - correction) / sigma_w

        return {
            "statistic": float(w_stat), 
            "p_value": float(scipy_p_val),
            "z_statistic": float(z_stat),
            "w_plus": float(w_plus),
            "w_minus": float(w_minus),
            "n_valid": int(n_valid)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------
# 2. BAĞIMLI (PAIRED) İKİ ÖRNEKLEM WILCOXON TESTİ
# ---------------------------------------------------------
class PairedWilcoxonRequest(BaseModel):
    data_1: List[float]  # Ön Test
    data_2: List[float]  # Son Test
    alternative: str = "two-sided" 
    conf_level: float = 0.95

@router.post("/paired-wilcoxon")
def paired_wilcoxon_test(request: PairedWilcoxonRequest):
    try:
        arr1 = np.array(request.data_1)
        arr2 = np.array(request.data_2)
        alt = request.alternative
        
        if len(arr1) != len(arr2):
            raise HTTPException(status_code=400, detail="Grupların gözlem sayıları birbirine eşit olmalıdır.")
            
        diffs = arr2 - arr1
        valid_diffs = diffs[diffs != 0]
        n_valid = len(valid_diffs)
        
        if n_valid < 1:
            raise HTTPException(status_code=400, detail="Farklı olan en az 1 gözlem çifti girmelisiniz.")
        
        abs_diffs = np.abs(valid_diffs)
        ranks = stats.rankdata(abs_diffs, method='average') 
        
        w_plus = np.sum(ranks[valid_diffs > 0])
        w_minus = np.sum(ranks[valid_diffs < 0])
        w_stat = min(w_plus, w_minus)
        
        try:
            res = stats.wilcoxon(arr2, arr1, alternative=alt, correction=True, exact=False)
            scipy_p_val = res.pvalue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SciPy Hesap Hatası: {str(e)}")

        mu_w = n_valid * (n_valid + 1) / 4.0
        unique_vals, counts = np.unique(abs_diffs, return_counts=True)
        tie_sum = np.sum(counts**3 - counts)
        var_w = (n_valid * (n_valid + 1) * (2 * n_valid + 1)) / 24.0 - tie_sum / 48.0
        sigma_w = np.sqrt(var_w) if var_w > 0 else 0
        
        z_stat = 0.0
        if sigma_w > 0:
            diff_w = w_plus - mu_w
            correction = 0.5 * np.sign(diff_w)
            if abs(diff_w) < 0.5:
                correction = diff_w
            z_stat = (diff_w - correction) / sigma_w

        return {
            "statistic": float(w_stat), 
            "p_value": float(scipy_p_val),
            "z_statistic": float(z_stat),
            "w_plus": float(w_plus),
            "w_minus": float(w_minus),
            "n_valid": int(n_valid)
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
