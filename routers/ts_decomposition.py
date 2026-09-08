from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose, STL

router = APIRouter(prefix="/test", tags=["Zaman Serisi Ayrıştırma"])

class DecomposeRequest(BaseModel):
    y: List[Optional[float]]
    period: int
    model_type: str = 'additive'  # 'additive', 'multiplicative' veya 'stl'

@router.post("/decomposition")
def calculate_decomposition(req: DecomposeRequest):
    try:
        series = pd.Series(req.y).interpolate() # Aradaki boşlukları doldur ki algoritma çökmesin
        n_obs = len(series)

        if n_obs < req.period * 2:
            raise ValueError(f"Ayrıştırma için veriniz çok kısa. En az 2 tam periyot ({req.period * 2} gözlem) gerekli.")

        if req.model_type == 'stl':
            stl = STL(series, period=req.period, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
        else:
            res = seasonal_decompose(series, model=req.model_type, period=req.period, extrapolate_trend='freq')
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid

        return {
            "trend": trend.fillna(0).tolist(),
            "seasonal": seasonal.fillna(0).tolist(),
            "residual": resid.fillna(0).tolist()
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ayrıştırma Hatası: {str(e)}")
