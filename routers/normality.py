from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from scipy.stats import shapiro, norm
import matplotlib
matplotlib.use('Agg') # Sunucunun çökmesini engellemek için arka plan çizim motorunu aktifleştirir
import matplotlib.pyplot as plt
import io
import base64

# Router'ı oluşturuyoruz
router = APIRouter(
    prefix="/test",
    tags=["Normallik Testleri"]
)

# Gelen verinin yapısı (Kullanıcıdan float tipinde bir liste bekliyoruz)
class DataInput(BaseModel):
    data: list[float]

@router.post("/shapiro")
async def perform_shapiro(input_data: DataInput):
    # Veriyi istatistik kütüphanelerinin okuyabileceği Numpy dizisine çeviriyoruz
    dataset = np.array(input_data.data)
    
    # 1. Shapiro-Wilk Testini Hesapla
    stat, p_value = shapiro(dataset)
    
    # 2. Matplotlib ile Grafiği Çiz (Histogram + Normal Dağılım Eğrisi)
    plt.figure(figsize=(8, 5))
    
    # Verinin histogramını çiz
    count, bins, ignored = plt.hist(dataset, bins='auto', density=True, alpha=0.6, color='steelblue', edgecolor='black', label='Veri Dağılımı')
    
    # İdeal Normal Dağılım çan eğrisini hesapla
    mu, std = dataset.mean(), dataset.std(ddof=1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    
    # Eğriyi grafiğe ekle
    plt.plot(x, p, 'k', linewidth=2, label=f'Normal Eğri (Ort={mu:.2f}, Std={std:.2f})')
    
    plt.title("Veri Dağılımı vs İdeal Normal Dağılım")
    plt.xlabel("Değerler")
    plt.ylabel("Yoğunluk")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Grafiği base64 koduna çevir (Frontend'e göndermek için)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close() # Belleği temizlemek için figürü kapat (Çok önemli)
    
    # 4. JSON olarak sonucu döndür
    return {
        "statistic": round(stat, 4),
        "p_value": round(p_value, 5),
        "is_normal": bool(p_value > 0.05),
        "interpretation": "Veri normal dağılıma uygundur." if p_value > 0.05 else "Veri normal dağılmamaktadır.",
        "plot_image": f"data:image/png;base64,{image_base64}"
    }
