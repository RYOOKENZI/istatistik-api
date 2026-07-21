from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

router = APIRouter(
    prefix="/test",
    tags=["Wilcoxon Testleri"]
)

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
        
        # 1. Farkları bul ve sıfırları çıkar
        diffs = arr - m0
        valid_diffs = diffs[diffs != 0]
        n_valid = len(valid_diffs)
        
        if n_valid < 1:
            raise HTTPException(status_code=400, detail="Test medyanına eşit olmayan en az 1 gözlem girmelisiniz.")
        
        # 2. Mutlak farkları bul ve SciPy'nin kusursuz rankdata'sını kullan
        abs_diffs = np.abs(valid_diffs)
        
        # SciPy'nin rankdata'sı eşitliklere (ties) ortalama rank verir (Örn: 1.5)
        ranks = stats.rankdata(abs_diffs, method='average') 
        
        # Pozitif ve Negatif Sıra Toplamlarını Bul (İşaretleme)
        w_plus = np.sum(ranks[valid_diffs > 0])
        w_minus = np.sum(ranks[valid_diffs < 0])
        
        # Geleneksel W istatistiği
        w_stat = min(w_plus, w_minus)
        
        # 3. Z ve P değerini doğrudan SciPy'nin core fonksiyonundan çekelim 
        # (Manuel hesaplama yerine hata payını sıfıra indirir)
        try:
            res = stats.wilcoxon(valid_diffs, alternative=alt, correction=True, exact=False)
            scipy_p_val = res.pvalue
            scipy_w_stat = res.statistic # SciPy'nin hesapladığı W
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SciPy Hesaplama Hatası: {str(e)}")

        # Etki büyüklüğü ve detaylar için Z istatistiğini bulalım (SciPy'den tersine mühendislik veya manuel)
        # exact=False olduğunda scipy.wilcoxon Z skorunu döndürmez, manuel bulalım:
        mu_w = n_valid * (n_valid + 1) / 4.0
        
        # Tie düzeltmeli Varyans
        unique_vals, counts = np.unique(abs_diffs, return_counts=True)
        tie_sum = np.sum(counts**3 - counts)
        var_w = (n_valid * (n_valid + 1) * (2 * n_valid + 1)) / 24.0 - tie_sum / 48.0
        sigma_w = np.sqrt(var_w) if var_w > 0 else 0
        
        z_stat = 0.0
        if sigma_w > 0:
            diff_w = w_plus - mu_w
            # Süreklilik düzeltmesi
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
