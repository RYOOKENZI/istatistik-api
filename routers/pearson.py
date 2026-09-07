from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import pingouin as pg
import numpy as np

router = APIRouter(prefix="/test", tags=["İlişki ve Korelasyon"])

# Frontend'in gönderdiği JSON verisinin şablonu
class PearsonRequest(BaseModel):
    data: Dict[str, List[float]]
    alpha: float = 0.05
    correction: str = "holm"  # "holm", "bonferroni" veya "none"

@router.post("/pearson")
def pearson_correlation(request: PearsonRequest):
    try:
        # 1. Veri Hazırlığı
        df = pd.DataFrame(request.data)
        
        # Geçerli sayısal sütunları seç
        num_df = df.select_dtypes(include=[np.number])
        cols = num_df.columns.tolist()
        
        if len(cols) < 2:
            raise HTTPException(status_code=400, detail="En az 2 sayısal değişken gereklidir.")
        
        # Listwise deletion (eksik verisi olan tüm satırları analizden çıkar)
        num_df = num_df.dropna()
        
        if len(num_df) < 3:
            raise HTTPException(status_code=400, detail="Analiz için en az 3 geçerli satır (gözlem) bulunmalıdır.")
            
        # Sabit (varyansı 0 olan) değişken kontrolü
        for col in cols:
            if num_df[col].var() == 0:
                raise HTTPException(status_code=400, detail=f"'{col}' değişkeninin varyansı sıfırdır. Korelasyon hesaplanamaz.")

        # 2. Pingouin ile Çoklu Korelasyon Matrisi ve Düzeltmeler
        corr_method = 'none' if request.correction == 'none' else request.correction
        
        # pairwise_corr, değişken çiftleri arasındaki korelasyonu ve p-değeri düzeltmelerini anında yapar
        pw = pg.pairwise_corr(num_df, method='pearson', padj=corr_method)
        
        results = []
        for index, row in pw.iterrows():
            # p-adj değeri (düzeltme yoksa raw p ile aynı olur)
            p_val_raw = float(row['p-unc'])
            p_val_adj = float(row['p-corr']) if 'p-corr' in row else p_val_raw
            
            # %95 Güven Aralığı sınırları
            ci_low, ci_up = float(row['CI95%'][0]), float(row['CI95%'][1])
            
            r = float(row['r'])
            n = int(row['n'])
            
            # Öğrenci T Testi İstatistiği (Bazı pingouin sürümleri t yerine BF10 verir, manuel güvenli hesap)
            t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
            
            # Fisher Z Dönüşümü ve Standart Hata (Frontend "Hesaplamaları Göster" sekmesi için gerekli)
            z_fisher = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
            se_z = 1 / np.sqrt(n - 3) if n > 3 else 0.0
            
            # Kovaryans Hesaplaması (SP_XY / N - Popülasyon formülü kullanılarak frontend ile eşlenir)
            cov_matrix = np.cov(num_df[row['X']], num_df[row['Y']], bias=True)
            cov_val = float(cov_matrix[0][1])

            results.append({
                "v1": str(row['X']),
                "v2": str(row['Y']),
                "n": n,
                "r": r,
                "r2": r**2,
                "t": float(t_stat),
                "p_raw": p_val_raw,
                "p_adj": p_val_adj,
                "ci_l": ci_low,
                "ci_u": ci_up,
                "z_fisher": float(z_fisher),
                "se_z": float(se_z),
                "cov": cov_val
            })

        return {
            "test": "Pearson Correlation",
            "pairwise": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Pearson Hatası: {str(e)}")
