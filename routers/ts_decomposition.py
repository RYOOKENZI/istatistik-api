# routers/ts_decomposition.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL

router = APIRouter(prefix="/test", tags=["Zaman Serisi Ayrıştırma"])

# HTML'in yollayabileceği her iki isme (model veya model_type) uyumlu hale getirdik
class DecomposeRequest(BaseModel):
    y: List[Optional[float]]
    period: int = 12
    model_type: str = 'additive'
    model: Optional[str] = None 

@router.post("/decomposition")
def calculate_decomposition(req: DecomposeRequest):
    try:
        # Verideki olası null değerleri (özellikle fark alınmış serilerde) güvenli şekilde doldur
        series = pd.Series(req.y).interpolate().bfill().ffill()
        n_obs = len(series)

        if n_obs < req.period * 2:
            raise ValueError(f"Ayrıştırma için veriniz çok kısa. En az 2 tam periyot ({req.period * 2} gözlem) gerekli.")

        # Hangi model isminin geldiğini algıla
        active_model = req.model if req.model else req.model_type

        # 1. AYRIŞTIRMA İŞLEMİ
        if active_model == 'stl':
            stl = STL(series, period=req.period, robust=True)
            res = stl.fit()
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid
        else:
            res = seasonal_decompose(series, model=active_model, period=req.period, extrapolate_trend='freq')
            trend = res.trend
            seasonal = res.seasonal
            resid = res.resid

        # 2. ÖZET İSTATİSTİKLER VE GÜÇ (STRENGTH) HESAPLAMALARI
        # HTML arayüzü muhtemelen Trendin ve Mevsimselliğin ne kadar "güçlü" olduğunu ekrana yazmak istiyor.
        var_resid = np.var(resid)
        var_trend_resid = np.var(trend + resid)
        var_seas_resid = np.var(seasonal + resid)
        
        # Trend Gücü: max(0, 1 - Var(R)/Var(T+R))
        trend_str = max(0, 1 - var_resid / var_trend_resid) if var_trend_resid > 0 else 0
        # Mevsimsellik Gücü: max(0, 1 - Var(R)/Var(S+R))
        seas_str = max(0, 1 - var_resid / var_seas_resid) if var_seas_resid > 0 else 0

        mean_resid = np.mean(resid)
        std_resid = np.std(resid)

        # 3. HTML'İN BEKLEDİĞİ YAPIYI DÖNDÜR
        return {
            "observed": series.tolist(),
            "trend": trend.fillna(0.0).tolist(),
            "seasonal": seasonal.fillna(0.0).tolist(),
            "residual": resid.fillna(0.0).tolist(),
            
            # Eğer HTML objenin içinde bekliyorsa (rRes.metrics.trend_strength gibi)
            "metrics": {
                "trend_strength": float(trend_str),
                "seasonal_strength": float(seas_str),
                "resid_mean": float(mean_resid),
                "resid_std": float(std_resid)
            },
            
            # Eğer HTML veriyi dışarıda (düz formatta) bekliyorsa (rRes.trend_strength gibi)
            "trend_strength": float(trend_str),
            "seasonal_strength": float(seas_str),
            "resid_mean": float(mean_resid),
            "resid_std": float(std_resid)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ayrıştırma Hatası: {str(e)}")
