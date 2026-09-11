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

        # Trend eşleştirmesi
        arch_trend = 'c'
        if req.trend == 'ct': arch_trend = 'ct'
        elif req.trend == 'nc': arch_trend = 'n' 

        # ADF/DFGLS için Lag parametresi (arch kütüphanesi KESİNLİKLE büyük/özel harf bekler)
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
            results["adf"] = {
                "stat": float(adf.stat), 
                "p_value": float(adf.pvalue), 
                "lag": int(adf.lags), 
                "crit": float(adf.critical_values.get('5%', 0))
            }
        except Exception as e: 
            results["adf"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        # 2. KPSS TESTİ
        try:
            kpss_trend = 'c' if arch_trend == 'n' else arch_trend
            kpss = KPSS(series, trend=kpss_trend)
            results["kpss"] = {
                "stat": float(kpss.stat), 
                "p_value": float(kpss.pvalue), 
                "lag": int(kpss.lags), 
                "crit": float(kpss.critical_values.get('5%', 0))
            }
        except Exception as e: 
            results["kpss"] = {"stat": 0, "p_value": 0.0, "lag": 0, "crit": 0}

        # 3. PHILLIPS-PERRON TESTİ
        try:
            pp = PhillipsPerron(series, trend=arch_trend)
            results["pp"] = {
                "stat": float(pp.stat), 
                "p_value": float(pp.pvalue), 
                "lag": int(pp.lags), 
                "crit": float(pp.critical_values.get('5%', 0))
            }
        except Exception as e: 
            results["pp"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        # 4. DF-GLS TESTİ
        try:
            gls_trend = 'c' if arch_trend == 'n' else arch_trend
            dfgls = DFGLS(series, trend=gls_trend, method=safe_method)
            
            stat_val = float(dfgls.stat)
            crit_5 = float(dfgls.critical_values.get('5%', 0))
            
            # arch kütüphanesinde DF-GLS için her zaman p-value yoktur. Varsa al, yoksa stat < crit mantığıyla üret:
            pval = getattr(dfgls, 'pvalue', None)
            if pval is None:
                pval = 0.001 if stat_val < crit_5 else 0.999
            else:
                pval = float(pval)
                
            results["dfgls"] = {
                "stat": stat_val, 
                "p_value": pval, 
                "lag": int(dfgls.lags), 
                "crit": crit_5
            }
        except Exception as e: 
            results["dfgls"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0}

        # 5. ZIVOT-ANDREWS TESTİ
        try:
            za_trend = arch_trend if arch_trend in ['c','t','ct'] else 'c'
            za = ZivotAndrews(series, trend=za_trend)
            
            stat_val = float(za.stat)
            crit_5 = float(za.critical_values.get('5%', 0))
            
            # arch kütüphanesinde Zivot-Andrews için KESİNLİKLE p-value yoktur. Stat ve Crit üzerinden mantıksal p-value üretiyoruz:
            pval = getattr(za, 'pvalue', None)
            if pval is None:
                pval = 0.001 if stat_val < crit_5 else 0.999
            else:
                pval = float(pval)

            results["za"] = {
                "stat": stat_val, 
                "p_value": pval, 
                "lag": int(za.lags), 
                "crit": crit_5,
                "break_idx": -1 # JS arayüzünde "Bilinmiyor" yazması için güvenli fallback
            }
        except Exception as e: 
            results["za"] = {"stat": 0, "p_value": 1.0, "lag": 0, "crit": 0, "break_idx": -1}

        return results

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Durağanlık Analizi Hatası: {str(e)}")
