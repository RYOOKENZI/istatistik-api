from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from scipy import stats

router = APIRouter(prefix="/test", tags=["Oran Testleri"])

# İstek (Request) Şeması
class BinomialRequest(BaseModel):
    successes: int       # Başarı sayısı (x)
    trials: int          # Toplam deneme (n)
    p_null: float = 0.5  # Hipotez Oranı (p0)
    alternative: str = "two-sided" 
    conf_level: float = 0.95

@router.post("/binomial")
def binomial_exact_test(request: BinomialRequest):
    try:
        k = request.successes
        n = request.trials
        p0 = request.p_null
        alt = request.alternative
        conf = request.conf_level

        # Mantıksal Hata Kontrolleri
        if k < 0 or n < 1 or k > n:
            raise HTTPException(status_code=400, detail="Geçersiz başarı veya deneme sayısı.")
        if not (0 < p0 < 1):
            raise HTTPException(status_code=400, detail="Hipotez oranı 0 ile 1 arasında olmalıdır.")

        # SciPy ile Kesin (Exact) Binom Testi
        # Not: stats.binomtest SciPy'nin modern ve en güncel binom fonksiyonudur.
        res = stats.binomtest(k, n, p=p0, alternative=alt)
        p_val = res.pvalue
        
        # Wilson Score Güven Aralığını SciPy üzerinden hesaplatma
        ci = res.proportion_ci(confidence_level=conf, method='wilson')

        return {
            "p_value": float(p_val),
            "ci_lower": float(ci.low),
            "ci_upper": float(ci.high),
            "observed_prop": float(k / n)
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SciPy Hesaplama Hatası: {str(e)}")
