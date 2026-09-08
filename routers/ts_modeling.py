# routers/ts_modeling.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

# ARIMA ve ETS kütüphaneleri
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

router = APIRouter(prefix="/test", tags=["Zaman Serisi Modelleme"])

class ModelingRequest(BaseModel):
    y: List[Optional[float]]
    model_type: str  # 'auto_arima', 'manual_arima', 'ets'
    params: Dict[str, Any] = {} # Kullanıcının girdiği p,d,q veya mevsimsellik parametreleri

@router.post("/modeling")
def calculate_model(req: ModelingRequest):
    try:
        series = pd.Series(req.y).dropna()
        n_obs = len(series)
        if n_obs < 10:
            raise ValueError("Modelleme için yetersiz veri.")

        result = {}

        # 1. AUTO-ARIMA (Yapay Zeka Destekli Seçim)
        if req.model_type == 'auto_arima':
            m = req.params.get('m', 1) # Mevsimsellik periyodu (Örn: 12)
            seasonal = m > 1
            
            # Auto-Arima'yı çalıştır (Hızlı arama modunda)
            model = pm.auto_arima(series, seasonal=seasonal, m=m, 
                                  stepwise=True, suppress_warnings=True, 
                                  error_action="ignore", trace=False)
            
            # Parametreleri al
            order = model.order
            seasonal_order = model.seasonal_order
            fitted = model.predict_in_sample()
            residuals = series - fitted

            name = f"ARIMA{order}"
            if seasonal:
                name = f"SARIMA{order}{seasonal_order}"

            result = {
                "name": name,
                "aic": float(model.aic()),
                "aicc": float(model.aicc()),
                "bic": float(model.bic()),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "fitted": fitted.tolist(),
                "residuals": residuals.tolist(),
                "order": order,
                "seasonal_order": seasonal_order
            }

        # 2. MANUEL ARIMA (Kullanıcının Seçtiği P, D, Q)
        elif req.model_type == 'manual_arima':
            p, d, q = req.params.get('p', 0), req.params.get('d', 0), req.params.get('q', 0)
            P, D, Q, m = req.params.get('P', 0), req.params.get('D', 0), req.params.get('Q', 0), req.params.get('m', 0)
            
            order = (p, d, q)
            seasonal_order = (P, D, Q, m) if m > 1 else (0, 0, 0, 0)
            
            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            res = model.fit()
            
            fitted = res.fittedvalues
            residuals = res.resid

            name = f"ARIMA{order}"
            if m > 1: name = f"SARIMA{order}{seasonal_order}"

            result = {
                "name": name,
                "aic": float(res.aic),
                "aicc": float(res.aicc) if hasattr(res, 'aicc') else float(res.aic),
                "bic": float(res.bic),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "fitted": fitted.tolist(),
                "residuals": residuals.tolist(),
                "order": order,
                "seasonal_order": seasonal_order
            }

        # 3. ETS (Üstel Düzleştirme)
        elif req.model_type == 'ets':
            trend = req.params.get('trend', 'add')
            seasonal = req.params.get('seasonal', 'add')
            m = req.params.get('m', 12)
            
            # ETS modeli sıfır veya negatif değerlerde 'mul' (çarpımsal) hata verebilir.
            if (seasonal == 'mul' or trend == 'mul') and (series <= 0).any():
                raise ValueError("Çarpımsal (Multiplicative) ETS modelleri için verinizde 0 veya negatif değer olmamalıdır.")

            model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=m if seasonal else None)
            res = model.fit()
            
            fitted = res.fittedvalues
            residuals = res.resid

            result = {
                "name": f"ETS({trend},{seasonal})",
                "aic": float(res.aic),
                "aicc": float(res.aicc),
                "bic": float(res.bic),
                "rmse": float(np.sqrt(np.mean(residuals**2))),
                "fitted": fitted.tolist(),
                "residuals": residuals.tolist()
            }
        
        else:
            raise ValueError("Bilinmeyen model türü.")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Modelleme Hatası: {str(e)}")
