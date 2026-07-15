from fastapi import APIRouter
from pydantic import BaseModel
from scipy import stats

router = APIRouter()

# Dışarıdan (HTML vitrininden) gelecek verinin formatı
class NormalityInput(BaseModel):
    veri: list[float]

@router.post("/shapiro-wilk")
def shapiro_wilk_test(data: NormalityInput):
    # Shapiro-Wilk testi matematiksel olarak en az 3 gözlem ister
    if len(data.veri) < 3:
        return {"hata": "Shapiro-Wilk testi için en az 3 gözlem gereklidir."}
    
    # SciPy ile hesaplama yapıyoruz
    stat, p_value = stats.shapiro(data.veri)
    
    # Sonuçları JSON formatında (sitenin anlayacağı dilde) paketliyoruz
    return {
        "test": "Shapiro-Wilk",
        "w_istatistigi": float(stat),
        "p_degeri": float(p_value),
        "normal_dagilim": bool(p_value > 0.05),
        "yorum": "Veriler normal dağılıma uygundur (p > 0.05)." if p_value > 0.05 else "Veriler normal dağılıma uymamaktadır (p <= 0.05)."
    }