from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats

router = APIRouter(
    prefix="/test",
    tags=["Variance"]
)

class LeveneRequest(BaseModel):
    groups: List[List[float]]
    center: str = 'median'
    alpha: float = 0.05

@router.post("/levene")
async def calculate_levene(request: LeveneRequest):
    # Verileri Numpy dizilerine çevir
    groups = [np.array(g) for g in request.groups]
    
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="En az 2 grup gereklidir.")
        
    k = len(groups)
    N = sum(len(g) for g in groups)
    
    try:
        # 1. LEVENE İSTATİSTİĞİ (Merkez seçimine göre)
        stat, p_val = stats.levene(*groups, center=request.center)
        
        # 2. KUSURSUZ SHAPIRO-WILK NORMALLİK TESTİ
        shapiro_p_values = []
        for g in groups:
            if len(g) >= 3:
                # Eğer gruptaki tüm sayılar aynıysa (varyans 0 ise) Shapiro hata verir!
                # Bunu engellemek için varyans kontrolü yapıyoruz.
                if np.var(g) == 0:
                    shapiro_p_values.append(1.0) # Tamamen aynı sayılar, teknik olarak normal değil ama çökmemesi için
                else:
                    stat_sw, p_sw = stats.shapiro(g)
                    shapiro_p_values.append(float(p_sw))
            else:
                shapiro_p_values.append(1.0) # 3 gözlemden az ise hesaplanamaz
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temel istatistik hatası: {str(e)}")

    # 3. ANOVA TABLOSU SAPMA HESAPLAMALARI
    Z_groups = []
    for g in groups:
        if request.center == 'mean':
            center_val = np.mean(g)
        elif request.center == 'median':
            center_val = np.median(g)
        else: # trimmed
            center_val = stats.trim_mean(g, 0.1)
        
        Z_groups.append(np.abs(g - center_val))
        
    Z_all = np.concatenate(Z_groups)
    grand_mean = np.mean(Z_all)
    
    SS_b = sum(len(zg) * (np.mean(zg) - grand_mean)**2 for zg in Z_groups)
    SS_w = sum(np.sum((zg - np.mean(zg))**2) for zg in Z_groups)
    
    df_b = k - 1
    df_w = N - k
    
    MS_b = SS_b / df_b if df_b > 0 else 0
    MS_w = SS_w / df_w if df_w > 0 else 0
    
    # 4. GERÇEK MATEMATİKSEL TEST GÜCÜ (POST-HOC POWER) HESAPLAMASI
    try:
        # Non-centrality parameter (lambda) hesaplaması
        f_stat = MS_b / MS_w if MS_w > 0 else 0
        nc_param = f_stat * df_b
        
        # Seçilen alpha değerine göre Kritik F noktasını bulma
        f_crit = stats.f.ppf(1 - request.alpha, df_b, df_w)
        
        # Merkezi olmayan (Non-central) F dağılımının kümülatif yoğunluk fonksiyonu üzerinden Güç (Power)
        power = 1.0 - stats.ncf.cdf(f_crit, df_b, df_w, nc_param)
        
        power = float(power)
        if np.isnan(power):
            power = 0.0
    except:
        power = 0.0 # Olası bir sıfıra bölünme hatasında yedek

    anova_table = {
        "df_b": int(df_b),
        "SS_b": float(SS_b),
        "MS_b": float(MS_b),
        "df_w": int(df_w),
        "SS_w": float(SS_w),
        "MS_w": float(MS_w),
        "df_t": int(N - 1),
        "SS_t": float(SS_b + SS_w)
    }

    # Çıktıya "power" eklendi!
    # routers/variance.py en altındaki return bloğu
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "anova_table": anova_table,
        "shapiro_p_values": [float(sp) for sp in shapiro_p_values],
        "power": float(power)  # Grafikteki sabit sayı kilidini açacak kritik değer
    }



from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import scipy.stats as stats

# Eğer main.py'de prefix="/test" olarak tanımlamadıysan, buraya prefix="/test" ekleyebilirsin.
# HTML dosyasındaki fetch isteği "/test/f-test" adresine gidiyor.
router = APIRouter(
    prefix="/test",
    tags=["Varyans Testleri"]
)

# Gelen JSON verisinin şeması
class FTestRequest(BaseModel):
    group1: list[float]
    group2: list[float]

@router.post("/f-test")
async def calculate_f_test(request: FTestRequest):
    g1 = np.array(request.group1)
    g2 = np.array(request.group2)
    
    # Güvenlik kontrolü: En az 2 gözlem olmalı
    if len(g1) < 2 or len(g2) < 2:
        raise HTTPException(status_code=400, detail="Her iki grup için en az 2'şer gözlem gereklidir.")
    
    # Yansız varyans tahmini için Bessel düzeltmesi (n-1) ile hesaplama (ddof=1)
    var1 = np.var(g1, ddof=1)
    var2 = np.var(g2, ddof=1)
    
    if var1 == 0 or var2 == 0:
        raise HTTPException(status_code=400, detail="Gruplardan birinin varyansı 0 olduğu için F istatistiği hesaplanamıyor.")
        
    # F İstatistiği (s1^2 / s2^2)
    f_statistic = var1 / var2
    
    # Serbestlik dereceleri (df = n - 1)
    df1 = len(g1) - 1
    df2 = len(g2) - 1
    
    # 2 Yönlü (2-tailed) P-değeri hesabı
    # F dağılımının kümülatif fonksiyonu (CDF) kullanılarak iki kuyruklu p-değeri bulunur
    cdf_value = stats.f.cdf(f_statistic, df1, df2)
    p_value = 2 * min(cdf_value, 1 - cdf_value)
    
    return {
        "statistic": float(f_statistic),
        "p_value": float(p_value)
    }
