from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats
from statsmodels.stats.power import tt_solve_power

# Router'ı tanımlıyoruz
router = APIRouter(
    prefix="/test",
    tags=["T-Testleri"]
)

# 1. Ön Yüzden Gelecek Veri Modeli (Request Schema)
class SummaryStats(BaseModel):
    n: int
    mean: float
    sd: float

class OneSampleTRequest(BaseModel):
    test_value: float
    alternative: str = "two-sided"  # 'two-sided', 'less', 'greater'
    conf_level: float = 0.95
    data_type: str  # 'ham' veya 'ozet'
    data: Optional[List[float]] = None
    summary: Optional[SummaryStats] = None
    remove_outliers: Optional[bool] = False

# 2. Endpoint'in Kendisi
@router.post("/one-sample-t")
def one_sample_t_test(request: OneSampleTRequest):
    try:
        mu0 = request.test_value
        alt = request.alternative
        alpha = 1 - request.conf_level
        
        # Scipy alternative parametresi eşleştirmesi
        if alt == "two-sided":
            scipy_alt = "two-sided"
        elif alt == "greater":
            scipy_alt = "greater"
        elif alt == "less":
            scipy_alt = "less"
        else:
            scipy_alt = "two-sided"

        # HAM VERİ MODU
        if request.data_type == "ham":
            if not request.data or len(request.data) < 2:
                raise HTTPException(status_code=400, detail="Ham veri modunda en az 2 gözlem gereklidir.")
            
            arr = np.array(request.data)
            
            # İsteğe bağlı aykırı değer temizliği (Basit IQR yöntemi)
            if request.remove_outliers:
                q1 = np.percentile(arr, 25)
                q3 = np.percentile(arr, 75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                arr = arr[(arr >= lower_bound) & (arr <= upper_bound)]
            
            n = len(arr)
            mean = np.mean(arr)
            sd = np.std(arr, ddof=1)
            
            # Scipy ile asıl testi yap
            res = stats.ttest_1samp(arr, popmean=mu0, alternative=scipy_alt)
            t_stat = res.statistic
            p_val = res.pvalue
            df = res.df
            
            plot_data = arr.tolist()

        # ÖZET İSTATİSTİK MODU
        elif request.data_type == "ozet":
            if not request.summary:
                raise HTTPException(status_code=400, detail="Özet istatistik verisi bulunamadı.")
            
            n = request.summary.n
            mean = request.summary.mean
            sd = request.summary.sd
            
            if n < 2 or sd < 0:
                raise HTTPException(status_code=400, detail="Geçersiz özet istatistik (n>=2, sd>=0 olmalı).")
            
            df = n - 1
            se = sd / np.sqrt(n)
            t_stat = (mean - mu0) / se
            
            # Manuel p-value hesaplama (özet veri için)
            if scipy_alt == "two-sided":
                p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
            elif scipy_alt == "greater":
                p_val = 1 - stats.t.cdf(t_stat, df)
            elif scipy_alt == "less":
                p_val = stats.t.cdf(t_stat, df)
                
            plot_data = None
            
        else:
            raise HTTPException(status_code=400, detail="Bilinmeyen data_type.")

        # Ortak Hesaplamalar (SE, Güven Aralığı, Kritik Değer, Test Gücü)
        se = sd / np.sqrt(n)
        
        # Kritik t Değeri
        if scipy_alt == "two-sided":
            t_crit = stats.t.ppf(1 - alpha/2, df)
        elif scipy_alt == "greater":
            t_crit = stats.t.ppf(1 - alpha, df)
        else: # less
            t_crit = stats.t.ppf(alpha, df) # Negatif değer döner

        # Güven Aralığı (Ortalama için)
        if scipy_alt == "two-sided":
            margin = stats.t.ppf(1 - alpha/2, df) * se
            ci_lower = mean - margin
            ci_upper = mean + margin
        elif scipy_alt == "greater":
            margin = stats.t.ppf(1 - alpha, df) * se
            ci_lower = mean - margin
            ci_upper = float('inf')
        else: # less
            margin = stats.t.ppf(1 - alpha, df) * se
            ci_lower = float('-inf')
            ci_upper = mean + margin

        # Test Gücü (Power) - Cohen's d (Etki Büyüklüğü) üzerinden
        cohens_d = (mean - mu0) / sd
        try:
            power_alt = scipy_alt if scipy_alt != "less" else "two-sided" # statsmodels less desteklemediği için yaklaşım
            power = tt_solve_power(effect_size=cohens_d, nobs=n, alpha=alpha, alternative=power_alt)
        except:
            power = 0.0

        # Sonucu JSON (Dict) olarak dön
        return {
            "statistic": float(t_stat),
            "p_value": float(p_val),
            "critical_value": float(t_crit),
            "conf_int": [float(ci_lower), float(ci_upper)],
            "power": float(power),
            "n": int(n),
            "mean": float(mean),
            "sd": float(sd),
            "se": float(se),
            "plot_data": plot_data
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
