from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import pingouin as pg
import scipy.stats as stats
import numpy as np

router = APIRouter(prefix="/test", tags=["İlişki ve Korelasyon"])

class KendallRequest(BaseModel):
    data: Dict[str, List[float]]
    alpha: float = 0.05
    correction: str = "holm"

@router.post("/kendall")
def kendall_correlation(request: KendallRequest):
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
                raise HTTPException(status_code=400, detail=f"'{col}' değişkeninin varyansı sıfırdır. Kendall Tau hesaplanamaz.")

        corr_method = 'none' if request.correction == 'none' else request.correction
        
        # Pingouin Kendall Tau-b method
        pw = pg.pairwise_corr(num_df, method='kendall', padj=corr_method)
        
        results = []
        for index, row in pw.iterrows():
            p_val_raw = float(row['p-unc'])
            p_val_adj = float(row['p-corr']) if 'p-corr' in row else p_val_raw
            
            # Pingouin normally returns CI for Pearson/Spearman, but Kendall CI might not be directly in the row
            ci_low, ci_up = 0.0, 0.0
            if 'CI95%' in row:
                ci_low, ci_up = float(row['CI95%'][0]), float(row['CI95%'][1])
                
            r = float(row['r'])
            n = int(row['n'])
            
            # Kendall için Manuel Fisher CI Fallback
            if ci_low == 0.0 and ci_up == 0.0:
                z_fisher = 0.5 * np.log((1 + r) / (1 - r + 1e-10))
                se_z = 0.437 / np.sqrt(n - 4) if n > 4 else 0.0
                z_crit = 1.96 if request.alpha == 0.05 else (2.576 if request.alpha == 0.01 else 1.645)
                z_low = z_fisher - z_crit * se_z
                z_up = z_fisher + z_crit * se_z
                ci_low = (np.exp(2*z_low) - 1) / (np.exp(2*z_low) + 1)
                ci_up = (np.exp(2*z_up) - 1) / (np.exp(2*z_up) + 1)
            
            results.append({
                "v1": str(row['X']),
                "v2": str(row['Y']),
                "n": n,
                "tau": r,
                "p_raw": p_val_raw,
                "p_adj": p_val_adj,
                "ci_l": float(ci_low),
                "ci_u": float(ci_up)
            })

        return {
            "test": "Kendall Tau-b Correlation Matrix",
            "pairwise": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Kendall Hatası: {str(e)}")
