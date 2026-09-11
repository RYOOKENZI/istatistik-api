from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

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
        # 1. PANDAS YERİNE SAF NUMPY KULLAN! (Arch kütüphanesi numpy ile kusursuz çalışır)
        raw_y = [val for val in req.y if val is not None]
        series = np.array(raw_y, dtype=float)
        n_obs = len(series)
        
        if n_obs < 15:
            raise ValueError(f"Durağanlık testleri için en az 15 geçerli gözlem gereklidir. Gelen Gözlem: {n_obs}")

        # Trend eşleştirmesi
        arch_trend = 'c'
        if req.trend == 'ct': arch_trend = 'ct'
        elif req.trend == 'nc': arch_trend = 'n' 

        # Güvenli Lag Metodu Eşleştirmesi
        l_method = req.lag_method.lower()
        if l_method == 'bic':
            safe_method = 'BIC'
        elif l_method == 't-stat':
            safe_method = 't-stat'
        else:
            safe_method = 'AIC'

        results = {}

        # 1. ADF TESTİ
        try:
            adf = ADF(series, trend=arch_trend, method=safe_method)
            results["adf"] = {"stat": float(adf.stat), "p_value": float(adf.pvalue), "lag": int(adf.lags), "crit": float(adf.critical_values.get('5%', 0))}
        except Exception as e: 
            raise ValueError(f"ADF Testi Hatası: {str(e)}")

        # 2. KPSS TESTİ
        try:
            kpss_trend = 'c' if arch_trend == 'n' else arch_trend
            kpss = KPSS(series, trend=kpss_trend)
            results["kpss"] = {"stat": float(kpss.stat), "p_value": float(kpss.pvalue), "lag": int(kpss.lags), "crit": float(kpss.critical_values.get('5%', 0))}
        except Exception as e: 
            raise ValueError(f"KPSS Testi Hatası: {str(e)}")

        # 3. PHILLIPS-PERRON TESTİ
        try:
            pp = PhillipsPerron(series, trend=arch_trend)
            results["pp"] = {"stat": float(pp.stat), "p_value": float(pp.pvalue), "lag": int(pp.lags), "crit": float(pp.critical_values.get('5%', 0))}
        except Exception as e: 
            raise ValueError(f"Phillips-Perron Testi Hatası: {str(e)}")

        # 4. DF-GLS TESTİ
        try:
            gls_trend = 'c' if arch_trend == 'n' else arch_trend
            dfgls = DFGLS(series, trend=gls_trend, method=safe_method)
            stat_val = float(dfgls.stat)
            crit_5 = float(dfgls.critical_values.get('5%', 0))
            
            # DF-GLS için her versiyonda p-value olmayabilir, stat < crit kontrolüyle manuel üretiyoruz
            pval = getattr(dfgls, 'pvalue', None)
            if pval is None: pval = 0.001 if stat_val < crit_5 else 0.999
                
            results["dfgls"] = {"stat": stat_val, "p_value": float(pval), "lag": int(dfgls.lags), "crit": crit_5}
        except Exception as e: 
            raise ValueError(f"DF-GLS Testi Hatası: {str(e)}")

        # 5. ZIVOT-ANDREWS TESTİ
        try:
            za_trend = arch_trend if arch_trend in ['c','t','ct'] else 'c'
            za = ZivotAndrews(series, trend=za_trend)
            stat_val = float(za.stat)
            crit_5 = float(za.critical_values.get('5%', 0))
            
            # Zivot-Andrews için her versiyonda p-value olmayabilir
            pval = getattr(za, 'pvalue', None)
            if pval is None: pval = 0.001 if stat_val < crit_5 else 0.999

            results["za"] = {"stat": stat_val, "p_value": float(pval), "lag": int(za.lags), "crit": crit_5, "break_idx": -1}
        except Exception as e: 
            raise ValueError(f"Zivot-Andrews Testi Hatası: {str(e)}")

        return results

    except Exception as e:
        # Sessiz hatayı (0.000) iptal ettik. Artık ön yüze tam hatayı bağıracak!
        raise HTTPException(status_code=500, detail=str(e))
