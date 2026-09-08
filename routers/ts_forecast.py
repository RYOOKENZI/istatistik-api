# routers/ts_forecast.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

router = APIRouter(prefix="/test", tags=["Zaman Serisi Tahmin"])

class ForecastRequest(BaseModel):
    y: List[Optional[float]]
    h: int = 12  # Kaç dönem ileri tahmin edilecek
    alpha: float = 0.05  # Güven aralığı (0.05 = %95)
    model_type: str # 'ARIMA', 'SARIMA', 'ETS'
    params: Dict[str, Any] = {} # order, seasonal_order vb.

@router.post("/forecast")
def calculate_forecast(req: ForecastRequest):
    try:
        series = pd.Series(req.y).dropna()
        n_obs = len(series)

        if n_obs < 5:
            raise ValueError("Tahmin yapabilmek için veri yetersiz.")

        # 1. ARIMA / SARIMA TAHMİNİ
        if 'ARIMA' in req.model_type:
            order = req.params.get('order', [0,0,0])
            seasonal_order = req.params.get('seasonal_order', [0,0,0,0])
            
            # Modeli tekrar kur ve eğit
            model = ARIMA(series, order=tuple(order), seasonal_order=tuple(seasonal_order))
            res = model.fit()

            # Tahmin ve Güven Aralıkları (get_forecast)
            forecast_obj = res.get_forecast(steps=req.h)
            point_forecast = forecast_obj.predicted_mean
            conf_int = forecast_obj.conf_int(alpha=req.alpha)

            return {
                "point_forecast": point_forecast.tolist(),
                "lower_bound": conf_int.iloc[:, 0].tolist(),
                "upper_bound": conf_int.iloc[:, 1].tolist()
            }

        # 2. ETS TAHMİNİ
        elif 'ETS' in req.model_type:
            trend = req.params.get('trend', 'add')
            seasonal = req.params.get('seasonal', 'add')
            m = req.params.get('m', 12)

            model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=m if seasonal else None)
            res = model.fit()

            # ETS Statsmodels'te varsayılan olarak güven aralığı vermez, simülasyonla veya standart sapmayla yaklaşık üretilir.
            point_forecast = res.forecast(req.h)
            
            # Basit Yaklaşık Güven Aralığı (Sadece Görsel UI için)
            resid_std = np.std(res.resid)
            z = 1.96 # %95
            lower_bound = point_forecast - (z * resid_std * np.sqrt(np.arange(1, req.h + 1)))
            upper_bound = point_forecast + (z * resid_std * np.sqrt(np.arange(1, req.h + 1)))

            return {
                "point_forecast": point_forecast.tolist(),
                "lower_bound": lower_bound.tolist(),
                "upper_bound": upper_bound.tolist()
            }

        else:
            raise ValueError("Geçersiz model türü gönderildi.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin Hatası: {str(e)}")
