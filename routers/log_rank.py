from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

router = APIRouter(prefix="/test", tags=["Sağkalım Analizleri"])

class LogRankRequest(BaseModel):
    time: List[float]
    event: List[float]
    group: List[str]  # Log-Rank için grup zorunludur
    conf_level: float = 0.95

@router.post("/log-rank")
def log_rank_test(request: LogRankRequest):
    try:
        # 1. Veri Hazırlığı
        df = pd.DataFrame({
            'time': request.time, 
            'event': request.event,
            'group': request.group
        })
        df = df.dropna()
        n = len(df)
        
        if n < 4:
            raise HTTPException(status_code=400, detail="Analiz için en az 4 geçerli gözlem gereklidir.")

        unique_groups = df['group'].unique()
        if len(unique_groups) < 2:
            raise HTTPException(status_code=400, detail="Log-Rank testi için en az 2 farklı grup gereklidir.")

        alpha = 1 - request.conf_level
        groups_dict = {}

        # 2. Genel Log-Rank Testi (Mantel-Cox)
        res = multivariate_logrank_test(df['time'], df['group'], df['event'])
        logrank_res = {
            "chi2": float(res.test_statistic),
            "df": int(res.degrees_of_freedom),
            "p_value": float(res.p_value)
        }
        
        # 3. İkili Çoklu Karşılaştırmalar (Pairwise Log-Rank) ve Holm Düzeltmesi
        pairwise_res = []
        if len(unique_groups) > 2:
            pw = pairwise_logrank_test(df['time'], df['group'], df['event'])
            pw_df = pw.summary
            
            raw_p_values = pw_df['p'].tolist()
            idx_pairs = pw_df.index.tolist()
            chi_stats = pw_df['test_statistic'].tolist()
            
            # Manuel Holm-Bonferroni Düzeltmesi
            m = len(raw_p_values)
            sorted_indices = np.argsort(raw_p_values)
            p_adj = np.zeros(m)
            
            for step, i in enumerate(sorted_indices):
                p_adj[i] = min(1.0, raw_p_values[i] * (m - step))
                if step > 0 and p_adj[i] < p_adj[sorted_indices[step-1]]:
                    p_adj[i] = p_adj[sorted_indices[step-1]]

            for i, pair in enumerate(idx_pairs):
                pairwise_res.append({
                    "g1": str(pair[0]),
                    "g2": str(pair[1]),
                    "chi2": float(chi_stats[i]),
                    "p_raw": float(raw_p_values[i]),
                    "p_adj": float(p_adj[i])
                })

        # 4. Her Grup İçin Kaplan-Meier Eğrisi ve Risk Tabloları
        for g in unique_groups:
            mask = df['group'] == g
            kmf = KaplanMeierFitter()
            kmf.fit(df['time'][mask], event_observed=df['event'][mask], alpha=alpha)
            
            event_table = kmf.event_table
            t_unq = event_table.index.tolist()
            n_risk = event_table['at_risk'].tolist()
            d_ev = event_table['observed'].tolist()
            c_cen = event_table['censored'].tolist()
            
            S = kmf.survival_function_.iloc[:, 0].tolist()
            CI = kmf.confidence_interval_
            CI_l = CI.iloc[:, 0].tolist()
            CI_u = CI.iloc[:, 1].tolist()
            
            median_surv = kmf.median_survival_time_
            median_str = float(median_surv) if not np.isinf(median_surv) else "Gözlenmedi"
            
            groups_dict[str(g)] = {
                "t_unq": [float(t) for t in t_unq],
                "n_risk": [int(x) for x in n_risk],
                "d_ev": [int(x) for x in d_ev],
                "c_cen": [int(x) for x in c_cen],
                "S": [float(x) for x in S],
                "CI_l": [float(x) for x in CI_l],
                "CI_u": [float(x) for x in CI_u],
                "median": median_str
            }
            
            # Frontend grafikleri için koordinatlar
            timeline = []
            for i, t in enumerate(t_unq):
                timeline.append({
                    "t": float(t), "s": float(S[i]), "ci_l": float(CI_l[i]), "ci_u": float(CI_u[i]),
                    "isCens": bool(c_cen[i] > 0 and d_ev[i] == 0)
                })
            groups_dict[str(g)]["timeline"] = timeline

        return {
            "method": "Log-Rank Test",
            "groups": groups_dict,
            "logrank": logrank_res,
            "pairwise": pairwise_res
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Log-Rank Hatası: {str(e)}")
