# routers/ts_forecast.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import re
from scipy.stats import norm

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

router = APIRouter(prefix="/test", tags=["Zaman Serisi Tahmin"])

# HTML'den gelen JSON paketine BİREBİR uyumlu model
class ForecastRequest(BaseModel):
    y: List[Optional[float]]
    model_type: str          # Örn: "ARIMA(1, 0, 1)" veya "ETS(add, add)"
    horizon: int = 12        # HTML 'horizon' olarak yolluyor
    conf_level: float = 0.95 # HTML 'conf_level' olarak yolluyor

@router.post("/forecast")
def calculate_forecast(req: ForecastRequest):
    try:
        series = pd.Series(req.y).dropna().reset_index(drop=True)
        n_obs = len(series)

        if n_obs < 5:
            raise ValueError("Tahmin yapabilmek için veri yetersiz.")

        model_name = req.model_type.upper()
        # Güven aralığını alfa seviyesine çevir (%95 -> 0.05)
        alpha_val = 1.0 - req.conf_level
        if alpha_val <= 0 or alpha_val >= 1:
            alpha_val = 0.05

        predictions, lower, upper = [], [], []
        fitted = np.full(n_obs, np.nan)

        # 1. ARIMA / SARIMA MODELLERİNİ PARÇALA VE TAHMİN ET
        if "ARIMA" in model_name:
            # Model isminden (Örn: SARIMA(1,0,1)(0,1,1,12)) sayıları otomatik ayıkla
            nums = re.findall(r'\d+', model_name)
            order = (0, 0, 0)
            seasonal_order = (0, 0, 0, 0)
            
            if len(nums) >= 3:
                order = (int(nums[0]), int(nums[1]), int(nums[2]))
            if len(nums) >= 7:
                seasonal_order = (int(nums[3]), int(nums[4]), int(nums[5]), int(nums[6]))

            model = ARIMA(series, order=order, seasonal_order=seasonal_order)
            res = model.fit()
            
            fitted = res.fittedvalues.values
            forecast_obj = res.get_forecast(steps=req.horizon)
            
            predictions = forecast_obj.predicted_mean.values
            conf_int = forecast_obj.conf_int(alpha=alpha_val)
            lower = conf_int.iloc[:, 0].values
            upper = conf_int.iloc[:, 1].values

        # 2. ETS (ÜSTEL DÜZLEŞTİRME) TAHMİNİ
        elif "ETS" in model_name:
            trend = 'add'
            seasonal = None
            
            if ',' in req.model_type:
                parts = req.model_type.replace('ETS(', '').replace(')', '').split(',')
                trend = parts[0].strip()
                if trend.lower() == 'none': trend = None
                if len(parts) > 1:
                    seasonal = parts[1].strip()
                    if seasonal.lower() == 'none': seasonal = None

            if (series <= 0).any():
                if trend == 'mul': trend = 'add'
                if seasonal == 'mul': seasonal = 'add'

            model = ExponentialSmoothing(series, trend=trend, seasonal=seasonal, seasonal_periods=12 if seasonal else None)
            res = model.fit()
            
            fitted = res.fittedvalues.values
            predictions = res.forecast(req.horizon).values
            
            # ETS için yaklaşik güven aralığı
            resid_std = np.std(res.resid)
            z = abs(norm.ppf(alpha_val / 2))
            margin = z * resid_std * np.sqrt(np.arange(1, req.horizon + 1))
            lower = predictions - margin
            upper = predictions + margin

        else:
            raise ValueError("Bilinmeyen model türü.")

        # 3. PERFORMANS METRİKLERİ (HTML'in sol altta beklediği RMSE, MAE, MAPE tabloları)
        y_arr = series.values
        # Sıfıra bölme hatasını önlemek için filtre
        valid_idx = (y_arr != 0) & (~np.isnan(fitted)) & (~np.isnan(y_arr))
        
        mae = np.mean(np.abs(y_arr - fitted)) if len(y_arr) > 0 else 0.0
        rmse = np.sqrt(np.mean((y_arr - fitted)**2)) if len(y_arr) > 0 else 0.0
        
        mape = 0.0
        smape = 0.0
        if valid_idx.sum() > 0:
            y_val = y_arr[valid_idx]
            f_val = fitted[valid_idx]
            mape = np.mean(np.abs((y_val - f_val) / y_val)) * 100
            
            denom = (np.abs(y_val) + np.abs(f_val)) / 2
            smape = np.mean(np.abs(y_val - f_val) / np.where(denom==0, 1e-8, denom)) * 100

        # 4. JSON DÖNÜŞÜ (HTML'deki rRes = { forecast: {...}, metrics: {...} } yapısının birebir aynısı)
        return {
            "forecast": {
                "predictions": predictions.tolist(),
                "lower": lower.tolist(),
                "upper": upper.tolist()
            },
            "metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "mape": float(mape),
                "smape": float(smape)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tahmin Hatası: {str(e)}")
