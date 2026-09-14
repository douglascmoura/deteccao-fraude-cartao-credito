"""
Módulo de treinamento e modelagem para detecção de fraudes.
Estrutura pipelines integrados com imbalanced-learn, evitando vazamento de dados no cross-validation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import StratifiedKFold

from src.metrics import evaluate_model


def get_model_candidates(scale_pos_weight: float = 10.0, random_state: int = 42) -> Dict[str, Any]:
    """
    Retorna os pipelines de modelos a serem comparados:
    1. Baseline sem balanceamento
    2. Under-sampling (RUS)
    3. Over-sampling (SMOTE)
    4. Ponderação de Classes (Class Weight / scale_pos_weight)
    5. Modelos de Conjunto (Random Forest e XGBoost)
    """
    models = {
        "1. Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "2. Logistic Regression (Under-sampling)": ImbPipeline([
            ("rus", RandomUnderSampler(random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "3. Logistic Regression (SMOTE)": ImbPipeline([
            ("smote", SMOTE(random_state=random_state)),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "4. Logistic Regression (Class-Weighted)": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state
        ),
        "5. Random Forest (Class-Weighted)": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced_subsample",
            random_state=random_state,
            n_jobs=-1
        ),
        "6. XGBoost (Cost-Sensitive)": XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1
        )
    }
    return models


def evaluate_candidates(
    models: Dict[str, Any],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Treina cada modelo candidato e avalia as predições no conjunto de teste intocado.
    
    Returns:
        df_results: Tabela comparativa com métricas de cada modelo.
        probabilities: Dicionário {nome_modelo: y_prob} para curvas ROC e PR.
        fitted_models: Dicionário de modelos treinados.
    """
    records = []
    probabilities = {}
    fitted_models = {}
    
    for name, model in models.items():
        print(f"[TREINANDO] {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        y_pred = model.predict(X_test)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            df_scores = model.decision_function(X_test)
            y_prob = (df_scores - df_scores.min()) / (df_scores.max() - df_scores.min())
        else:
            y_prob = None
            
        if y_prob is not None:
            probabilities[name] = y_prob
            
        metrics = evaluate_model(y_test.values, y_pred, y_prob)
        metrics["Modelo"] = name
        records.append(metrics)
        
    df_results = pd.DataFrame(records)
    
    cols_order = [
        "Modelo", "Recall", "Precision", "F1-Score", "PR-AUC", "ROC-AUC",
        "TP", "FP", "FN", "TN"
    ]
    df_results.rename(columns={
        "True Positives (TP)": "TP",
        "False Positives (FP)": "FP",
        "False Negatives (FN)": "FN",
        "True Negatives (TN)": "TN"
    }, inplace=True)
    
    existing_cols = [c for c in cols_order if c in df_results.columns]
    df_results = df_results[existing_cols].sort_values(by="PR-AUC", ascending=False).reset_index(drop=True)
    
    return df_results, probabilities, fitted_models
