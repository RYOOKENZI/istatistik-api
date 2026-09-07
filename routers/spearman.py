from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import pingouin as pg
import numpy as np

router = APIRouter(prefix="/test", tags=["İlişki ve Korelasyon"])

class SpearmanRequest(BaseModel):
    data: Dict[str, List[float]]
    alpha: float = 0.05
    correction: str = "holm"

@router.post("/spearman")
def spearman_correlation(request: SpearmanRequest):
    try:
        df = pd.DataFrame(request.data)
        
        num_df = df.select_dtypes(include=[np.number])
        cols = num_df.columns.tolist()
        
        if len(cols) < 2:
            raise HTTPException(status_code=400, detail="En az 2 sayısal değişken gereklidir.")
        
        num_df = num_df.dropna()
        if len(num_df) < 3:
            raise HTTPException(status_code=400, detail="Analiz için en az 3 geçerli satır (gözlem) bulunmalıdır.")
            
        for col in cols:
            if num_df[col].var() == 0:
                raise HTTPException(status_code=400, detail=f"'{col}' değişkeninin varyansı sıfırdır. Spearman korelasyonu hesaplanamaz.")

        corr_method = 'none' if request.correction == 'none' else request.correction
        
        # Pingouin Spearman method ('spearman' applies pearson to ranks automatically dealing with ties)
        pw = pg.pairwise_corr(num_df, method='spearman', padj=corr_method)
        
        results = []
        for index, row in pw.iterrows():
            p_val_raw = float(row['p-unc'])
            p_val_adj = float(row['p-corr']) if 'p-corr' in row else p_val_raw
            
            ci_low, ci_up = float(row['CI95%'][0]), float(row['CI95%'][1])
            r = float(row['r'])
            n = int(row['n'])
            
            t_stat = r * np.sqrt((n - 2) / (1 - r**2 + 1e-10))
            
            z_fisher = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
            se_z = 1.06 / np.sqrt(n - 3) if n > 3 else 0.0
            
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
                "se_z": float(se_z)
            })

        return {
            "test": "Spearman Rank Correlation Matrix",
            "pairwise": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Spearman Hatası: {str(e)}")
