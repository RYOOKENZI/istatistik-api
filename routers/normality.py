from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from scipy.stats import shapiro, norm, kstest
import matplotlib
matplotlib.use('Agg') # Sunucu çökmesini önlemek için
import matplotlib.pyplot as plt
import io
import base64

# Router'ı oluşturuyoruz
router = APIRouter(
    prefix="/test",
    tags=["Normallik Testleri"]
)

# --- MODELLER ---
class DataInputShapiro(BaseModel):
    data: list[float]

class DataInputKS(BaseModel):
    data: list[float]
    test_type: str  # "ks" veya "lilliefors"

# ==========================================
# 1. SHAPIRO-WILK TESTİ
# ==========================================
@router.post("/shapiro")
async def perform_shapiro(input_data: DataInputShapiro):
    dataset = np.array(input_data.data)
    stat, p_value = shapiro(dataset)
    
    plt.figure(figsize=(8, 5))
    plt.hist(dataset, bins='auto', density=True, alpha=0.6, color='steelblue', edgecolor='black')
    
    mu, std = dataset.mean(), dataset.std(ddof=1)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.plot(x, norm.pdf(x, mu, std), 'k', linewidth=2)
    
    plt.title("Shapiro-Wilk: Veri vs Normal Dağılım")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    
    return {
        "statistic": round(stat, 4),
        "p_value": round(p_value, 5),
        "is_normal": bool(p_value > 0.05),
        "plot_image": f"data:image/png;base64,{image_base64}"
    }

# ==========================================
# 2. K-S VE LILLIEFORS TESTİ
# ==========================================
@router.post("/ks-lilliefors")
async def perform_ks_lilliefors(input_data: DataInputKS):
    dataset = np.array(input_data.data)
    test_type = input_data.test_type.lower()
    n = len(dataset)
    mu, std = dataset.mean(), dataset.std(ddof=1)
    
    if test_type == "lilliefors":
        from statsmodels.stats.diagnostic import lilliefors
        stat, p_value = lilliefors(dataset, dist='norm', pvalmethod='approx')
        test_name = "Lilliefors Düzeltmeli K-S Testi"
    else:
        stat, p_value = kstest(dataset, 'norm', args=(mu, std))
        test_name = "Standart Kolmogorov-Smirnov Testi"
        
    plt.figure(figsize=(8, 5))
    
    # ECDF
    x_sorted = np.sort(dataset)
    y_ecdf = np.arange(1, n + 1) / n
    plt.step(x_sorted, y_ecdf, label='ECDF', where='post', color='steelblue', linewidth=2)
    
    # Teorik CDF
    x_theo = np.linspace(min(x_sorted) - std, max(x_sorted) + std, 200)
    y_theo = norm.cdf(x_theo, mu, std)
    plt.plot(x_theo, y_theo, label='Teorik CDF', color='red', linestyle='--', linewidth=2)
    
    plt.title(f"{test_name} Karşılaştırması")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
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


@router.post("/ks-lilliefors")
async def perform_ks_lilliefors(input_data: DataInputKS):
    dataset = np.array(input_data.data)
    test_type = input_data.test_type.lower()
    n = len(dataset)
    
    # Parametreleri veriden tahmin ediyoruz
    mu, std = dataset.mean(), dataset.std(ddof=1)
    
    # EĞER OTOMATİK İSE VE NORMAL DAĞILIM İSE -> LILLIEFORS KULLAN
    if test_type == "lilliefors" or (test_type == "auto"):
        from statsmodels.stats.diagnostic import lilliefors
        # Lilliefors, parametrelerin veriden tahmin edildiğini bilir ve düzeltmesini yapar
        stat, p_value = lilliefors(dataset, dist='norm', pvalmethod='approx')
        test_name = "Lilliefors Düzeltmeli K-S Testi"
    else:
        # Standart K-S (Parametreler dışarıdan veriliyorsa)
        stat, p_value = kstest(dataset, 'norm', args=(mu, std))
        test_name = "Standart Kolmogorov-Smirnov Testi (Dikkat: Parametreler veriden tahmin edildi)"
    
    # ... grafik çizim kodları aynı ...
