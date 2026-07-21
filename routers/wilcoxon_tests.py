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
    test_value: float # Test edilecek M0 değeri
    alternative: str = "two-sided"  # 'two-sided', 'less', 'greater'
    conf_level: float = 0.95
    data: List[float] # Sadece ham veri

@router.post("/one-sample-wilcoxon")
def one_sample_wilcoxon_test(request: OneSampleWilcoxonRequest):
    try:
        arr = np.array(request.data)
        m0 = request.test_value
        alt = request.alternative
        
        # 1. Farkları bul ve M0'a tam eşit olanları (sıfır fark) analizden çıkar
        diffs = arr - m0
        valid_diffs = diffs[diffs != 0]
        n_valid = len(valid_diffs)
        
        if n_valid < 1:
            raise HTTPException(status_code=400, detail="Test medyanına eşit olmayan en az 1 gözlem girmelisiniz.")
        
        # 2. SciPy ile Wilcoxon Testi 
        # exact=False -> Büyük örneklemler için Z yaklaşımını kullanır
        # correction=True -> Z yaklaşımı için süreklilik (continuity) düzeltmesi yapar
        try:
            res = stats.wilcoxon(valid_diffs, alternative=alt, correction=True, exact=False)
            scipy_p_val = res.pvalue
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"SciPy Hesaplama Hatası: {str(e)}")
        
        # 3. İleri Düzey Matematiksel Çıktılar İçin (W+, W-, Z, Ties) Manuel Hesaplamalar
        abs_diffs = np.abs(valid_diffs)
        ranks = stats.rankdata(abs_diffs) # Eşit değerlere (ties) ortalama rank verir
        
        w_plus = np.sum(ranks[valid_diffs > 0])
        w_minus = np.sum(ranks[valid_diffs < 0])
        w_stat = min(w_plus, w_minus) # Geleneksel W istatistiği
        
        # Beklenen Değer (mu_w) ve Tie (Eşitlik) Düzeltmeli Varyans (var_w)
        mu_w = n_valid * (n_valid + 1) / 4.0
        
        unique_vals, counts = np.unique(abs_diffs, return_counts=True)
        tie_sum = np.sum(counts**3 - counts)
        
        var_w = (n_valid * (n_valid + 1) * (2 * n_valid + 1)) / 24.0 - tie_sum / 48.0
        sigma_w = np.sqrt(var_w) if var_w > 0 else 0
        
        # Z-İstatistiği Hesaplama (Süreklilik Düzeltmesi ile)
        z_stat = 0.0
        if sigma_w > 0:
            diff_w = w_plus - mu_w
            correction = 0.5 * np.sign(diff_w)
            # Düzeltmenin farkın yönünü değiştirmesini engelle
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
