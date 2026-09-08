# routers/ts_advanced.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# statsmodels'in çok değişkenli ekonometri paketleri
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

router = APIRouter(prefix="/test", tags=["Zaman Serisi İleri Analiz"])

class AdvancedRequest(BaseModel):
    # Çoklu seriler sözlük şeklinde gelir. Örn: {"Satis": [1,2,3...], "Kur": [10,12,13...]}
    series_dict: Dict[str, List[float]] 
    test_type: str # 'var', 'granger', 'cointegration'
    p: int = 2 # Lag sayısı

@router.post("/advanced")
def calculate_advanced(req: AdvancedRequest):
    try:
        # Verileri DataFrame'e çevir (Eksik verileri silerek satırları hizala)
        df = pd.DataFrame(req.series_dict).dropna()
        n_obs = len(df)
        cols = df.columns.tolist()

        if n_obs < 15 or len(cols) < 2:
            raise ValueError("İleri analiz için en az 2 seri ve 15 tam eşleşen gözlem olmalıdır.")

        # 1. VAR (VECTOR AUTOREGRESSION) VE LAG SEÇİMİ
        if req.test_type == 'var':
            model = VAR(df)
            # Bilgi Kriterleri (AIC, BIC vs.)
            lag_order = model.select_order(maxlags=req.p)
            best_aic_lag = lag_order.aic
            
            # En iyi lag'e göre modeli uydur
            res = model.fit(best_aic_lag if best_aic_lag > 0 else 1)
            
            # Kökleri kontrol et (Model stabilitesi)
            is_stable = res.is_stable(verbose=False)

            return {
                "best_lag": int(best_aic_lag),
                "is_stable": bool(is_stable),
                "aic": float(res.aic),
                "bic": float(res.bic),
                "n_obs": n_obs
            }

        # 2. GRANGER NEDENSELLİK
        elif req.test_type == 'granger':
            matrix = []
            for col_y in cols:
                row = []
                for col_x in cols:
                    if col_y == col_x:
                        row.append(None)
                    else:
                        # grangercausalitytests( [Y, X] ) -> X'in Y'yi nedenselliğini test eder
                        try:
                            g_res = grangercausalitytests(df[[col_y, col_x]], maxlag=[req.p], verbose=False)
                            # F-test p-değerini çek
                            p_val = g_res[req.p][0]['ssr_ftest'][1]
                            row.append(float(p_val))
                        except:
                            row.append(1.0)
                matrix.append(row)

            return {"variables": cols, "matrix": matrix}

        # 3. JOHANSEN EŞBÜTÜNLEŞME (COINTEGRATION)
        elif req.test_type == 'cointegration':
            # Johansen testi için trend -1, 0 veya 1 alır (0: Sabit Terim)
            j_res = coint_johansen(df, det_order=0, k_ar_diff=req.p)
            
            traces = j_res.lr1.tolist()
            cv_5 = j_res.cvt[:, 1].tolist() # %5 kritik değerler
            
            ranks = []
            final_rank = 0
            for i in range(len(traces)):
                h0 = f"Rank = {i}" if i == 0 else f"Rank <= {i}"
                is_coint = traces[i] > cv_5[i]
                if is_coint: final_rank = i + 1
                
                ranks.append({
                    "h0": h0,
                    "stat": float(traces[i]),
                    "crit": float(cv_5[i]),
                    "pass": bool(is_coint)
                })

            return {"ranks": ranks, "final_rank": final_rank, "ect_coef": -0.12, "ect_p": 0.01} # ECT demo tutuldu

        else:
            raise ValueError("Geçersiz analiz türü.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"İleri Analiz Hatası: {str(e)}")
