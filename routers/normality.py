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



from pydantic import BaseModel
import numpy as np
from scipy.stats import shapiro, norm, kstest
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# K-S ve Lilliefors için API'ye gelecek verinin modeli
class DataInputKS(BaseModel):
    data: list[float]
    test_type: str  # "ks" veya "lilliefors" olarak arayüzden gelecek

# ... (Mevcut shapiro rotan burada kalsın) ...

@router.post("/ks-lilliefors")
async def perform_ks_lilliefors(input_data: DataInputKS):
    dataset = np.array(input_data.data)
    test_type = input_data.test_type.lower()
    n = len(dataset)
    
    # Parametreleri örneklemden hesapla
    mu, std = dataset.mean(), dataset.std(ddof=1)
    
    if test_type == "lilliefors":
        # Lilliefors Testi (statsmodels kütüphanesinden)
        from statsmodels.stats.diagnostic import lilliefors
        stat, p_value = lilliefors(dataset, dist='norm', pvalmethod='approx')
        test_name = "Lilliefors Düzeltmeli K-S Testi"
    else:
        # Standart Kolmogorov-Smirnov Testi
        stat, p_value = kstest(dataset, 'norm', args=(mu, std))
        test_name = "Standart Kolmogorov-Smirnov Testi"
        
    # --- K-S GRAFİĞİ: ECDF vs Normal CDF Çizimi ---
    plt.figure(figsize=(8, 5))
    
    # Ampirik Yığmalı Dağılım (ECDF)
    x_sorted = np.sort(dataset)
    y_ecdf = np.arange(1, n + 1) / n
    plt.step(x_sorted, y_ecdf, label='Ampirik Dağılım (ECDF)', where='post', color='steelblue', linewidth=2)
    
    # Teorik Normal Yığmalı Dağılım (CDF)
    x_theo = np.linspace(min(x_sorted) - std, max(x_sorted) + std, 200)
    y_theo = norm.cdf(x_theo, mu, std)
    plt.plot(x_theo, y_theo, label=f'Teorik CDF ($N({mu:.2f}, {std:.2f}^2)$)', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"{test_name} Karşılaştırması")
    plt.xlabel("Değerler")
    plt.ylabel("Yığmalı Olasılık")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Grafiği Base64'e çevir
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        "test_name": test_name,
        "statistic": round(stat, 4),
        "p_value": round(p_value, 5),
        "is_normal": bool(p_value > 0.05),
        "plot_image": f"data:image/png;base64,{image_base64}"
    }
