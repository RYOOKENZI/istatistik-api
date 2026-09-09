# routers/ts_decomposition.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL

router = APIRouter(prefix="/test", tags=["Zaman Serisi Ayrıştırma"])

class DecomposeRequest(BaseModel):
    y: List[Optional[float]]
    period: int = 12
    model_type: str = 'additive'
    model: Optional[str] = None 
    robust: Optional[bool] = False # Arayüz 'robust' parametresini de yolluyor

@router.post("/decomposition")
def calculate_decomposition(req: DecomposeRequest):
    try:
        series = pd.Series(req.y).interpolate().bfill().ffill()
        n_obs = len(series)

        if n_obs < req.period * 2:
            raise ValueError(f"Ayrıştırma için veriniz çok kısa. En az 2 tam periyot ({req.period * 2} gözlem) gerekli.")

        active_model = req.model if req.model else req.model_type

        # 1. AYRIŞTIRMA İŞLEMİ (STL VEYA KLASİK)
        if active_model == 'stl' or active_model == 'additive' and req.model == 'stl':
            # STL modeli (Robust parametresiyle)
            stl = STL(series, period=req.period, robust=req.robust)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
            active_model = 'additive' # STL her zaman toplamsaldır
        else:
            # Klasik Ayrıştırma
            res = seasonal_decompose(series, model=active_model, period=req.period, extrapolate_trend='freq')
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid

        # 2. MEVSİMSEL İNDEKS HESAPLAMASI (HTML'in çöktüğü yeri düzeltiyoruz)
        # Sadece ilk tam döngüyü (period kadar olan kısmı) almak indeksleri verir
        seasonal_list = seasonal.fillna(0.0).tolist()
        seasonal_indices = []
        
        for i in range(req.period):
            if i < len(seasonal_list):
                seasonal_indices.append(float(seasonal_list[i]))
            else:
                seasonal_indices.append(0.0 if active_model == 'additive' else 1.0)

        # 3. ÖZET İSTATİSTİKLER VE GÜÇ (STRENGTH)
        var_resid = np.var(resid)
        var_trend_resid = np.var(trend + resid)
        var_seas_resid = np.var(seasonal + resid)
        
        trend_str = float(max(0, 1 - var_resid / var_trend_resid)) if var_trend_resid > 0 else 0.0
        seas_str = float(max(0, 1 - var_resid / var_seas_resid)) if var_seas_resid > 0 else 0.0

        # 4. JSON DÖNÜŞÜ (Tam olarak HTML'in beklediği format)
        return {
            "observed": series.tolist(),
            "trend": trend.fillna(0.0).tolist(),
            "seasonal": seasonal_list,
            "residual": resid.fillna(0.0).tolist(),
            "seasonal_indices": seasonal_indices,    # İŞTE EKSİK OLAN BUYDU!
            "trend_strength": trend_str,
            "seasonal_strength": seas_str
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ayrıştırma Hatası: {str(e)}")
