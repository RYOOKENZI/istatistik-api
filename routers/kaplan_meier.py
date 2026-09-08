from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, pairwise_logrank_test

router = APIRouter(prefix="/test", tags=["Sağkalım Analizleri"])

class KaplanMeierRequest(BaseModel):
    time: List[float]
    event: List[float]
    group: Optional[List[str]] = None
    conf_level: float = 0.95

@router.post("/kaplan-meier")
def kaplan_meier_test(request: KaplanMeierRequest):
    try:
        # DataFrame Oluşturma
        df_dict = {'time': request.time, 'event': request.event}
        has_group = False
        if request.group and len(request.group) == len(request.time):
            df_dict['group'] = request.group
            has_group = True
            
        df = pd.DataFrame(df_dict)
        df = df.dropna()
        n = len(df)
        
        if n < 3:
            raise HTTPException(status_code=400, detail="Analiz için en az 3 geçerli gözlem gereklidir.")

        alpha = 1 - request.conf_level
        groups_dict = {}

        if has_group:
            unique_groups = df['group'].unique()
            
            # Overall Log-Rank (Multivariate) if > 1 group
            logrank_res = None
            if len(unique_groups) > 1:
                res = multivariate_logrank_test(df['time'], df['group'], df['event'])
                logrank_res = {
                    "chi2": float(res.test_statistic),
                    "df": int(res.degrees_of_freedom),
                    "p_value": float(res.p_value)
                }
                
            # Pairwise Log-Rank with Holm correction
            pairwise_res = []
            if len(unique_groups) > 2:
                pw = pairwise_logrank_test(df['time'], df['group'], df['event']) # Default is no correction in simple call, but summary table has p
                # Lifelines pairwise returns a summary dataframe
                pw_df = pw.summary
                
                # Apply Holm manually since lifelines default pairwise doesn't expose it cleanly in dictionary
                raw_p_values = pw_df['p'].tolist()
                idx_pairs = pw_df.index.tolist()
                chi_stats = pw_df['test_statistic'].tolist()
                
                # Holm correction
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

            for g in unique_groups:
                mask = df['group'] == g
                kmf = KaplanMeierFitter()
                kmf.fit(df['time'][mask], event_observed=df['event'][mask], alpha=alpha)
                
                # Timeline logic
                event_table = kmf.event_table
                t_unq = event_table.index.tolist()
                n_risk = event_table['at_risk'].tolist()
                d_ev = event_table['observed'].tolist()
                c_cen = event_table['censored'].tolist()
                
                S = kmf.survival_function_.iloc[:, 0].tolist()
                CI = kmf.confidence_interval_
                CI_l = CI.iloc[:, 0].tolist()
                CI_u = CI.iloc[:, 1].tolist()
                
                # SE is roughly implied by Greenwood, lifelines hides it deep, we pass CI
                groups_dict[str(g)] = {
                    "t_unq": [float(t) for t in t_unq],
                    "n_risk": [int(n) for n in n_risk],
                    "d_ev": [int(d) for d in d_ev],
                    "c_cen": [int(c) for c in c_cen],
                    "S": [float(s) for s in S],
                    "SE": [0.0] * len(S), # Frontend JS calculates SE for display anyway
                    "CI_l": [float(c) for c in CI_l],
                    "CI_u": [float(c) for c in CI_u],
                    "median": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else "Gözlenmedi",
                    "median_ci": [float(kmf.median_survival_time_), float(kmf.median_survival_time_)] # Optional feature extraction
                }
                
                # Create timeline objects for frontend chart
                timeline = []
                for i, t in enumerate(t_unq):
                    timeline.append({
                        "t": float(t), "s": float(S[i]), "ci_l": float(CI_l[i]), "ci_u": float(CI_u[i]),
                        "isCens": bool(c_cen[i] > 0 and d_ev[i] == 0)
                    })
                groups_dict[str(g)]["timeline"] = timeline

            return {
                "method": "Kaplan-Meier",
                "groups": groups_dict,
                "logrank": logrank_res,
                "pairwise": pairwise_res
            }

        else:
            # Single Group
            kmf = KaplanMeierFitter()
            kmf.fit(df['time'], event_observed=df['event'], alpha=alpha)
            
            event_table = kmf.event_table
            S = kmf.survival_function_.iloc[:, 0].tolist()
            CI = kmf.confidence_interval_
            
            timeline = []
            for i, t in enumerate(event_table.index):
                timeline.append({
                    "t": float(t), "s": float(S[i]), "ci_l": float(CI.iloc[i, 0]), "ci_u": float(CI.iloc[i, 1]),
                    "isCens": bool(event_table['censored'].iloc[i] > 0 and event_table['observed'].iloc[i] == 0)
                })

            groups_dict["Tüm Veri"] = {
                "t_unq": [float(t) for t in event_table.index],
                "n_risk": [int(n) for n in event_table['at_risk']],
                "d_ev": [int(d) for d in event_table['observed']],
                "c_cen": [int(c) for c in event_table['censored']],
                "S": [float(s) for s in S],
                "SE": [0.0] * len(S),
                "CI_l": [float(c) for c in CI.iloc[:, 0]],
                "CI_u": [float(c) for c in CI.iloc[:, 1]],
                "median": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else "Gözlenmedi",
                "timeline": timeline
            }

            return {
                "method": "Kaplan-Meier",
                "groups": groups_dict,
                "median_survival": float(kmf.median_survival_time_) if not np.isinf(kmf.median_survival_time_) else None
            }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Kaplan-Meier Hatası: {str(e)}")
