from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Union
import numpy as np
from scipy import stats

router = APIRouter(prefix="/test", tags=["Non-Parametrik Testler"])

class MannWhitneyRequest(BaseModel):
    group_1: List[float]
    group_2: List[float]
    alternative: str = "two-sided"
    continuity: bool = True

@router.post("/mann-whitney")
def mann_whitney_test(request: MannWhitneyRequest):
    try:
        arr1 = np.array(request.group_1, dtype=float)
        arr2 = np.array(request.group_2, dtype=float)
        alt = request.alternative
        
        n1, n2 = len(arr1), len(arr2)
        N = n1 + n2
        
        if n1 < 2 or n2 < 2:
            raise HTTPException(status_code=400, detail="Her grupta en az 2 veri olmalıdır.")
            
        # 1. Asimptotik P-değeri (Büyük örneklemler için hızlı hesaplama)
        res_asymp = stats.mannwhitneyu(arr1, arr2, use_continuity=request.continuity, alternative=alt, method="asymptotic")
        u_stat = float(res_asymp.statistic) # SciPy 1. grubun U değerini döndürür (U1)
        asymp_p = float(res_asymp.pvalue)

        # 2. Kesin (Exact) P-değeri Hesaplaması
        # Sadece N <= 50 ise hesaplatıyoruz (Kombinasyon hesaplaması çok büyük N'lerde sunucuyu dondurabilir)
        exact_p = "Hesaplanamadı (N > 50)"
        if N <= 50:
            try:
                res_exact = stats.mannwhitneyu(arr1, arr2, use_continuity=request.continuity, alternative=alt, method="exact")
                exact_p = float(res_exact.pvalue)
            except:
                pass
        
        # 3. Z İstatistiği Hesabı (SciPy MWU'da Z'yi direkt vermediği için manuel hesaplıyoruz)
        all_data = np.concatenate((arr1, arr2))
        _, counts = np.unique(all_data, return_counts=True)
        ties_sum = np.sum(counts**3 - counts)
        
        mu_u = n1 * n2 / 2.0
        var_u = (n1 * n2 / 12.0) * ((N + 1) - ties_sum / (N * (N - 1)))
        sigma_u = np.sqrt(var_u) if var_u > 0 else 1e-10
        
        u_adj = u_stat
        if request.continuity and u_stat != mu_u:
            u_adj = u_stat - 0.5 if u_stat > mu_u else u_stat + 0.5
            
        z_stat = (u_adj - mu_u) / sigma_u
        
        # 4. Etki Büyüklükleri (Effect Sizes)
        r_effect = abs(z_stat) / np.sqrt(N)
        rank_biserial = 1 - (2 * u_stat) / (n1 * n2)
        
        # Cliff's Delta = (U1 - U2) / (n1 * n2)
        u2 = (n1 * n2) - u_stat
        cliffs_delta = (u_stat - u2) / (n1 * n2)

        # Karar verilen nihai p-değeri (Mümkünse Kesin p, değilse Asimptotik p)
        final_p = asymp_p if type(exact_p) == str else exact_p

        return {
            "u_statistic": float(u_stat),
            "z_statistic": float(z_stat),
            "p_value": float(final_p),
            "asymp_p": float(asymp_p),
            "exact_p": exact_p,
            "r_effect": float(r_effect),
            "rank_biserial": float(rank_biserial),
            "cliffs_delta": float(cliffs_delta)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python U-Testi Hatası: {str(e)}")
