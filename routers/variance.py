from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

# Eğer bunu ayrı bir dosyaya (örn: routers/variance.py) koyacaksan:
router = APIRouter()
# Eğer doğrudan main.py içine koyacaksan "router" yerine "app" kullanmalısın (örn: @app.post)

class LeveneRequest(BaseModel):
    groups: List[List[float]]
    center: str = "median"
    alpha: float = 0.05

@router.post("/test/levene")
async def calculate_levene(request: LeveneRequest):
    if len(request.groups) < 2:
        raise HTTPException(status_code=400, detail="En az 2 grup gereklidir.")
    
    center_method = request.center if request.center in ["mean", "median", "trimmed"] else "median"

    try:
        shapiro_p_values = []
        for g in request.groups:
            if len(g) >= 3:
                stat, p_val = stats.shapiro(g)
                shapiro_p_values.append(float(p_val))
            else:
                shapiro_p_values.append(1.0)

        levene_stat, levene_p = stats.levene(*request.groups, center=center_method)

        k = len(request.groups)
        N = sum(len(g) for g in request.groups)
        z_groups = []
        for g in request.groups:
            arr = np.array(g)
            center_val = np.mean(arr) if center_method == "mean" else np.median(arr) if center_method == "median" else stats.trim_mean(arr, 0.05)
            z_groups.append(np.abs(arr - center_val))
            
        z_grand_mean = np.concatenate(z_groups).mean()
        ss_b = sum(len(z) * (np.mean(z) - z_grand_mean)**2 for z in z_groups)
        ss_w = sum(np.sum((z - np.mean(z))**2) for z in z_groups)

        return {
            "statistic": float(levene_stat),
            "p_value": float(levene_p),
            "shapiro_p_values": shapiro_p_values,
            "anova_table": {
                "SS_b": float(ss_b), "df_b": int(k - 1), "MS_b": float(ss_b / (k-1) if (k-1)>0 else 0),
                "SS_w": float(ss_w), "df_w": int(N - k), "MS_w": float(ss_w / (N-k) if (N-k)>0 else 0),
                "SS_t": float(ss_b + ss_w), "df_t": int(N - 1)
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))