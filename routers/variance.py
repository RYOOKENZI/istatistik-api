from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import numpy as np
from scipy import stats

# Router tanımı (Frontend'deki /test/levene url'sini oluşturur)
router = APIRouter(
    prefix="/test",
    tags=["Variance"]
)

# Frontend'den gelecek JSON verisinin modeli
class LeveneRequest(BaseModel):
    groups: List[List[float]]
    center: str = 'median'
    alpha: float = 0.05

@router.post("/levene")
async def calculate_levene(request: LeveneRequest):
    # Verileri Numpy dizilerine çevir
    groups = [np.array(g) for g in request.groups]
    
    # En az 2 grup kontrolü
    if len(groups) < 2:
        raise HTTPException(status_code=400, detail="En az 2 grup gereklidir.")
        
    k = len(groups)
    N = sum(len(g) for g in groups)
    
    try:
        # 1. SCIPY İLE ASIL LEVENE İSTATİSTİĞİ
        # Frontend 'median', 'mean' veya 'trimmed' gönderebilir
        stat, p_val = stats.levene(*groups, center=request.center)
        
        # Her grup için Shapiro-Wilk normallik testi
        shapiro_p_values = []
        for g in groups:
            if len(g) >= 3:  # Shapiro-Wilk için en az 3 gözlem gerekir
                _, sp = stats.shapiro(g)
                shapiro_p_values.append(sp)
            else:
                shapiro_p_values.append(1.0)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İstatistiksel hesaplama hatası: {str(e)}")

    # 2. ANOVA TABLOSU İÇİN SAPMA HESAPLAMALARI
    # Levene testinin doğası gereği, merkezden olan mutlak sapmaları (Z) buluyoruz
    Z_groups = []
    for g in groups:
        if request.center == 'mean':
            center_val = np.mean(g)
        elif request.center == 'median':
            center_val = np.median(g)
        else: # trimmed
            center_val = stats.trim_mean(g, 0.1) # %10 trim
        
        Z_groups.append(np.abs(g - center_val))
        
    # Z değerlerinin genel ortalaması (Grand Mean)
    Z_all = np.concatenate(Z_groups)
    grand_mean = np.mean(Z_all)
    
    # Kareler Toplamı Hesapları (Sum of Squares)
    SS_b = sum(len(zg) * (np.mean(zg) - grand_mean)**2 for zg in Z_groups)
    SS_w = sum(np.sum((zg - np.mean(zg))**2) for zg in Z_groups)
    
    # Serbestlik Dereceleri (Degrees of Freedom)
    df_b = k - 1
    df_w = N - k
    
    # Ortalama Kareler (Mean Squares)
    MS_b = SS_b / df_b if df_b > 0 else 0
    MS_w = SS_w / df_w if df_w > 0 else 0
    
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

    # Frontend'in beklediği tam JSON yapısı
    return {
        "statistic": float(stat),
        "p_value": float(p_val),
        "anova_table": anova_table,
        "shapiro_p_values": [float(sp) for sp in shapiro_p_values]
    }
