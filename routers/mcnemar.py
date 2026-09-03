from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import scipy.stats as stats
import numpy as np

router = APIRouter(prefix="/test", tags=["Kategorik Testler"])

class McNemarBowkerRequest(BaseModel):
    matrix: List[List[int]]
    k: int
    method: Optional[str] = "auto"

@router.post("/mcnemar-bowker")
def mcnemar_bowker_test(request: McNemarBowkerRequest):
    try:
        mat = np.array(request.matrix)
        k = request.k
        
        # Matris boyut kontrolü
        if mat.shape != (k, k):
            raise HTTPException(status_code=400, detail="Matrix boyutları uyumsuz.")
            
        chi2 = 0.0
        df = k * (k - 1) / 2
        exact_p = "Hesaplanamadı"
        used_method = request.method if request.method else "auto"

        # 1. KLASİK McNEMAR (2x2)
        if k == 2:
            b = int(mat[0][1])
            c = int(mat[1][0])
            m = b + c
            
            if m == 0:
                raise HTTPException(status_code=400, detail="b+c toplamı 0 olduğunda test hesaplanamaz.")

            # Otomatik yöntem seçimi
            if used_method == "auto":
                used_method = "exact" if m < 25 else "yates"

            # Kesin (Exact) Binom P-Değeri (Büyük N değerlerinde sunucuyu korumak için max 150 sınırı)
            if m <= 150:
                exact_p = float(stats.binomtest(min(b, c), m, p=0.5, alternative='two-sided').pvalue)

            if used_method == "classic":
                chi2 = float((b - c)**2 / m)
                p_val = float(stats.distributions.chi2.sf(chi2, 1))
                used_method = "Klasik McNemar"
            elif used_method == "yates":
                chi2 = float((max(0, abs(b - c) - 1))**2 / m)
                p_val = float(stats.distributions.chi2.sf(chi2, 1))
                used_method = "McNemar (Yates Düzeltmeli)"
            else: # exact
                chi2 = float((b - c)**2 / m)
                p_val = exact_p if isinstance(exact_p, float) else float(stats.distributions.chi2.sf((max(0, abs(b - c) - 1))**2 / m, 1))
                used_method = "Exact McNemar (Kesin Binom)"
                
        # 2. McNEMAR-BOWKER İÇSEL SİMETRİ TESTİ (k > 2)
        else:
            for i in range(k):
                for j in range(i + 1, k):
                    nij = int(mat[i][j])
                    nji = int(mat[j][i])
                    if nij + nji > 0:
                        chi2 += float((nij - nji)**2 / (nij + nji))
            
            p_val = float(stats.distributions.chi2.sf(chi2, df))
            used_method = "McNemar-Bowker Testi"

        return {
            "chi_square": chi2,
            "p_value": p_val,
            "exact_p": exact_p,
            "method_used": used_method
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python McNemar Hatası: {str(e)}")
