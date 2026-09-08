from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd

from arch.unitroot import ADF, KPSS, PhillipsPerron, DFGLS, ZivotAndrews

router = APIRouter(prefix="/test", tags=["Zaman Serisi Durağanlık"])

class StationarityRequest(BaseModel):
    y: List[Optional[float]]
    trend: str = 'c'
    alpha: float = 0.05
    lag_method: str = 'AIC'

@router.post("/stationarity_advanced")
def calculate_stationarity(req: StationarityRequest):
    try:
        series = pd.Series(req.y).dropna()
        n_obs = len(series)
        
        if n_obs < 15:
            raise ValueError("Durağanlık testleri için en az 15 geçerli gözlem gereklidir.")

        arch_trend = 'c'
        if req.trend == 'ct': arch_trend = 'ct'
        elif req.trend == 'nc': arch_trend = 'n' 

        results = {}

        # 1. ADF
        try:
            adf = ADF(series, trend=arch_trend, method=req.lag_method)
            results["adf"] = {"stat": float(adf.stat), "p_value": float(adf.pvalue), "lag": int(adf.lags), "crit": float(adf.critical_values.get('5%', 0))}
        except: results["adf"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        # 2. KPSS
        try:
            kpss_trend = 'c' if arch_trend == 'n' else arch_trend
            kpss = KPSS(series, trend=kpss_trend)
            results["kpss"] = {"stat": float(kpss.stat), "p_value": float(kpss.pvalue), "lag": int(kpss.lags), "crit": float(kpss.critical_values.get('5%', 0))}
        except: results["kpss"] = {"stat": 0, "p_value": 0.0, "lag": 0, "crit": 0}

        # 3. Phillips-Perron
        try:
            pp = PhillipsPerron(series, trend=arch_trend)
            results["pp"] = {"stat": float(pp.stat), "p_value": float(pp.pvalue), "lag": int(pp.lags), "crit": float(pp.critical_values.get('5%', 0))}
        except: results["pp"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        # 4. Zivot-Andrews
        try:
            za = ZivotAndrews(series, trend=arch_trend if arch_trend in ['c','t','ct'] else 'c')
            results["za"] = {"stat": float(za.stat), "p_value": float(za.pvalue), "lag": int(za.lags), "crit": float(za.critical_values.get('5%', 0))}
        except: results["za"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Durağanlık Analizi Hatası: {str(e)}")
