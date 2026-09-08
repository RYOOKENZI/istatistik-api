from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import roc_curve, roc_auc_score

router = APIRouter(prefix="/test", tags=["Regresyon Modelleri"])

class LogisticRegressionRequest(BaseModel):
    x: Dict[str, List[float]] # Birden fazla X kolonu
    y: List[float] # Binary (0,1)
    alpha: float = 0.05
    conf_level: float = 0.95

@router.post("/logistic-regression")
def logistic_regression_test(request: LogisticRegressionRequest):
    try:
        # DataFrame Oluşturma
        df_dict = {'Y_Bagimli': request.y}
        x_names = list(request.x.keys())
        for xn in x_names:
            df_dict[xn] = request.x[xn]
            
        df = pd.DataFrame(df_dict)
        df = df.dropna()
        n = len(df)
        k = len(x_names)
        
        if n < k + 2:
            raise HTTPException(status_code=400, detail=f"Gözlem sayısı (n={n}), bağımsız değişken sayısından (k={k}) yetersizdir. MLE çalıştırılamıyor.")
            
        unique_y = df['Y_Bagimli'].unique()
        if len(unique_y) != 2:
            raise HTTPException(status_code=400, detail="Bağımlı değişken Y yalnızca 0 ve 1 gibi iki sınıf (kategori) içermelidir.")

        # Lojistik Regresyon (Logit Model)
        X_df = df[x_names]
        X = sm.add_constant(X_df)
        
        # Olası ayrışma (separation) problemleri için catch-block
        try:
            model = sm.Logit(df['Y_Bagimli'], X).fit(disp=0)
        except Exception as mle_err:
            raise HTTPException(status_code=400, detail="Model yakınsamadı (Perfect Separation veya Multicollinearity). Lütfen bağımsız değişkenlerinizi kontrol edin.")

        # Temel İstatistikler
        prsq = float(model.prsquared) if hasattr(model, 'prsquared') and not np.isnan(model.prsquared) else 0.0
        llr = float(model.llr) if hasattr(model, 'llr') and not np.isnan(model.llr) else 0.0
        llr_p = float(model.llr_pvalue) if hasattr(model, 'llr_pvalue') and not np.isnan(model.llr_pvalue) else 1.0
        aic = float(model.aic)
        bic = float(model.bic)
        
        # Güven Aralıkları ve Odds Ratio
        conf_int = model.conf_int(alpha=1 - request.conf_level)
        
        coefs = []
        odds_ratios = []
        
        # Sabit (Intercept)
        coefs.append({
            "name": "Intercept",
            "estimate": float(model.params.iloc[0]),
            "std_error": float(model.bse.iloc[0]),
            "z": float(model.tvalues.iloc[0]), # Logit'te z-değeridir
            "p_value": float(model.pvalues.iloc[0]),
            "ci_lower": float(conf_int.iloc[0, 0]),
            "ci_upper": float(conf_int.iloc[0, 1])
        })
        odds_ratios.append({
            "estimate": float(np.exp(model.params.iloc[0])),
            "ci_lower": float(np.exp(conf_int.iloc[0, 0])),
            "ci_upper": float(np.exp(conf_int.iloc[0, 1]))
        })

        # X Değişkenleri
        for xn in x_names:
            coefs.append({
                "name": xn,
                "estimate": float(model.params[xn]),
                "std_error": float(model.bse[xn]),
                "z": float(model.tvalues[xn]),
                "p_value": float(model.pvalues[xn]),
                "ci_lower": float(conf_int.loc[xn, 0]),
                "ci_upper": float(conf_int.loc[xn, 1])
            })
            odds_ratios.append({
                "estimate": float(np.exp(model.params[xn])),
                "ci_lower": float(np.exp(conf_int.loc[xn, 0])),
                "ci_upper": float(np.exp(conf_int.loc[xn, 1]))
            })

        # Tahminler ve Artıklar (Deviance Residuals lojistikte varsayılandır)
        predictions = model.predict(X).tolist()
        resid_dev = model.resid_dev.tolist()
        
        # Etkili Gözlemler (Lojistikte Cook's D için OLS Yaklaşımı)
        # Statsmodels native influence object for logit is limited, fallback to leverage approximation
        try:
            influence = model.get_influence()
            cooks_d = influence.cooks_distance[0].tolist()
        except:
            cooks_d = [0] * n # Fail-safe

        # VIF Hesaplama
        vif_data = []
        tolerance_data = []
        if k > 1:
            for i in range(1, X.shape[1]): 
                vif_val = variance_inflation_factor(X.values, i)
                vif_data.append(float(vif_val))
                tolerance_data.append(1/float(vif_val) if vif_val != 0 else 0)

        # ROC - AUC
        try:
            auc = float(roc_auc_score(df['Y_Bagimli'], predictions))
            fpr, tpr, _ = roc_curve(df['Y_Bagimli'], predictions)
        except:
            auc = 0.5
            fpr, tpr = [0,1], [0,1]

        return {
            "method": "Logistic Regression",
            "n": n,
            "pseudo_r2": prsq,
            "lr_statistic": llr,
            "lr_p_value": llr_p,
            "aic": aic,
            "bic": bic,
            "roc_auc": auc,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "coefficients": coefs,
            "odds_ratios": odds_ratios,
            "vif": vif_data,
            "tolerance": tolerance_data,
            "predictions": {"probabilities": predictions},
            "residuals": {"deviance": resid_dev},
            "diagnostics": {"cooks_d": cooks_d}
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Python Lojistik Regresyon Hatası: {str(e)}")
