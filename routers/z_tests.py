from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats

router = APIRouter(
    prefix="/test",
    tags=["Z-Testleri"]
)

# Ön Yüzden Gelecek Özet Veri Şeması
class ZSummaryStats(BaseModel):
    n: int
    mean: float
    sd: float  # Z testinde bu değer popülasyon standart sapması (sigma) olarak kullanılacak

# Ön Yüzden Gelecek Ana Şema
class OneSampleZRequest(BaseModel):
    test_value: float
    pop_std: float
    alternative: str = "two-sided"  # 'two-sided', 'less', 'greater'
    conf_level: float = 0.95
    data_type: str  # 'ham' veya 'ozet'
    data: Optional[List[float]] = None
    summary: Optional[ZSummaryStats] = None
    remove_outliers: Optional[bool] = False

@router.post("/one-sample-z")
def one_sample_z_test(request: OneSampleZRequest):
    try:
        mu0 = request.test_value
        sigma = request.pop_std
        alt = request.alternative
        alpha = 1 - request.conf_level
        
        if sigma <= 0:
            raise HTTPException(status_code=400, detail="Popülasyon standart sapması (sigma) 0'dan büyük olmalıdır.")

        # HAM VERİ MODU
        if request.data_type == "ham":
            if not request.data or len(request.data) < 2:
                raise HTTPException(status_code=400, detail="En az 2 gözlem gereklidir.")
            
            arr = np.array(request.data)
            n = len(arr)
            mean = np.mean(arr)
            plot_data = arr.tolist()

        # ÖZET İSTATİSTİK MODU
        elif request.data_type == "ozet":
            if not request.summary:
                raise HTTPException(status_code=400, detail="Özet istatistik verisi bulunamadı.")
            
            n = request.summary.n
            mean = request.summary.mean
            
            if n < 2:
                raise HTTPException(status_code=400, detail="Gözlem sayısı (n) en az 2 olmalıdır.")
            
            plot_data = None
            
        else:
            raise HTTPException(status_code=400, detail="Bilinmeyen data_type.")

        # 1. Z İstatistiği Hesaplama
        se = sigma / np.sqrt(n)
        z_stat = (mean - mu0) / se

        # 2. P-Değeri Hesaplama (Standart Normal Dağılım üzerinden)
        if alt == "two-sided":
            p_val = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        elif alt == "greater":
            p_val = 1 - stats.norm.cdf(z_stat)
        elif alt == "less":
            p_val = stats.norm.cdf(z_stat)
        else:
            raise HTTPException(status_code=400, detail="Geçersiz alternatif hipotez yönü.")

        # 3. Kritik Z Değeri Hesaplama
        if alt == "two-sided":
            z_crit = stats.norm.ppf(1 - alpha/2)
        elif alt == "greater":
            z_crit = stats.norm.ppf(1 - alpha)
        else: # less
            z_crit = stats.norm.ppf(alpha)

        # 4. Güven Aralığı Hesaplama (Z Dağılımı ile)
        if alt == "two-sided":
            margin = stats.norm.ppf(1 - alpha/2) * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        elif alt == "greater":
            margin = stats.norm.ppf(1 - alpha) * se
            ci_lower = mean - margin
            ci_upper = float('inf')
        else: # less
            margin = stats.norm.ppf(1 - alpha) * se
            ci_lower = float('-inf')
            ci_upper = mean + margin

        # 5. İstatistiksel Güç (Power) Hesaplama
        cohens_d = (mean - mu0) / sigma
        ncp = abs(cohens_d) * np.sqrt(n) # Non-centrality parameter
        
        if alt == "two-sided":
            z_alpha_half = stats.norm.ppf(1 - alpha/2)
            power = 1 - stats.norm.cdf(z_alpha_half - ncp) + stats.norm.cdf(-z_alpha_half - ncp)
        else:
            z_alpha = stats.norm.ppf(1 - alpha)
            power = 1 - stats.norm.cdf(z_alpha - ncp)

        return {
            "statistic": float(z_stat),
            "p_value": float(p_val),
            "critical_value": float(z_crit),
            "conf_int": [float(ci_lower), float(ci_upper)],
            "power": float(power),
            "n": int(n),
            "mean": float(mean),
            "sd": float(sigma), # Z testinde örneklem standart sapması yerine sigma raporlanır
            "se": float(se),
            "plot_data": plot_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
