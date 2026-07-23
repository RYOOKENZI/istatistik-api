from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
import numpy as np
from scipy import stats

router = APIRouter(prefix="/test", tags=["İşaret Testleri"])

# İstek (Request) Şeması
class SignTestRequest(BaseModel):
    data_1: List[float]  # Ön Test / 1. Grup (X)
    data_2: List[float]  # Son Test / 2. Grup (Y)
    alternative: str = "two-sided" 
    conf_level: float = 0.95

@router.post("/sign-test")
def paired_sign_test(request: SignTestRequest):
    try:
        arr1 = np.array(request.data_1)
        arr2 = np.array(request.data_2)
        alt = request.alternative
        
        # 1. HATA KONTROLÜ: Gözlem sayıları eşit mi?
        if len(arr1) != len(arr2):
            raise HTTPException(status_code=400, detail="Grupların gözlem sayıları birbirine eşit olmalıdır.")
            
        # 2. FARKLARIN HESAPLANMASI (d_i = y_i - x_i)
        diffs = arr2 - arr1
        
        # Sıfır farklar (Ties/Eşitlikler) analizden atılır
        valid_diffs = diffs[diffs != 0]
        n_valid = len(valid_diffs)
        
        if n_valid < 1:
            raise HTTPException(status_code=400, detail="Farklı olan en az 1 gözlem çifti girmelisiniz.")
        
        # 3. İŞARET SAYIMLARI (Frekanslar)
        n_pos = int(np.sum(valid_diffs > 0))
        n_neg = int(np.sum(valid_diffs < 0))
        n_zero = int(len(diffs) - n_valid)
        
        # 4. TEST İSTATİSTİĞİ VE KESİN (EXACT) P-DEĞERİ HESABI
        # SciPy'nin modern binomtest motoru p=0.5 olasılığıyla çalıştırılır
        if alt == "two-sided":
            k_stat = min(n_pos, n_neg)
            res = stats.binomtest(n_pos, n_valid, p=0.5, alternative="two-sided")
        elif alt == "greater":
            # Son testin Ön testten büyük olma (Pozitiflerin baskınlığı) durumu
            k_stat = n_pos
            res = stats.binomtest(n_pos, n_valid, p=0.5, alternative="greater")
        elif alt == "less":
            # Son testin Ön testten küçük olma (Negatiflerin baskınlığı) durumu
            k_stat = n_neg
            res = stats.binomtest(n_pos, n_valid, p=0.5, alternative="less")
        else:
            raise HTTPException(status_code=400, detail="Geçersiz test yönü parametresi.")
            
        p_val = res.pvalue

        # 5. NORMAL DAĞILIM YAKLAŞIMLI Z-DEĞERİ (Süreklilik Düzeltmeli)
        # n >= 10 durumlarında literatürde Z istatistiği de raporlanır
        z_stat = 0.0
        if n_valid > 0:
            diff_from_expected = abs(n_pos - 0.5 * n_valid)
            corrected_diff = max(0.0, diff_from_expected - 0.5)
            z_stat = corrected_diff / (0.5 * np.sqrt(n_valid))
            if n_pos < 0.5 * n_valid:
                z_stat = -z_stat

        return {
            "statistic": float(k_stat), 
            "p_value": float(p_val),
            "z_statistic": float(z_stat),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_ties": n_zero,
            "n_valid": n_valid
        }
        
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SciPy Hesaplama Hatası: {str(e)}")
