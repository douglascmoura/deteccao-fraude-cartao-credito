"""
Módulo para análise e tratamento conservador de outliers via IQR.
Projetado especificamente para problemas de fraude onde instâncias raras não devem ser descartadas indevidamente.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional


def detect_outliers_iqr(
    df: pd.DataFrame,
    features: List[str],
    factor: float = 2.5,
    target_col: str = "Class"
) -> Dict[str, dict]:
    """
    Analisa os limites inferior e superior de cada feature via IQR e contabiliza outliers por classe.
    
    Args:
        df: DataFrame contendo as features e opcionalmente o target.
        features: Lista de nomes de colunas a analisar.
        factor: Multiplicador do IQR (2.5 é conservador).
        target_col: Coluna da classe alvo.
        
    Returns:
        Dicionário com estatísticas de corte por variável.
    """
    summary = {}
    
    for feat in features:
        q25 = df[feat].quantile(0.25)
        q75 = df[feat].quantile(0.75)
        iqr = q75 - q25
        cutoff = iqr * factor
        lower_bound = q25 - cutoff
        upper_bound = q75 + cutoff
        
        is_outlier = (df[feat] < lower_bound) | (df[feat] > upper_bound)
        total_outliers = is_outlier.sum()
        
        stat = {
            "q25": q25,
            "q75": q75,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "total_outliers": int(total_outliers)
        }
        
        if target_col in df.columns:
            stat["outliers_fraud"] = int((is_outlier & (df[target_col] == 1)).sum())
            stat["outliers_legit"] = int((is_outlier & (df[target_col] == 0)).sum())
            
        summary[feat] = stat
        
    return summary


def remove_extreme_outliers(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    features: List[str],
    factor: float = 2.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove ruídos extremos EXCLUSIVAMENTE do conjunto de treino.
    
    IMPORTANTE: Em problemas de fraude, as fraudes naturalmente ocupam as caudas das distribuições.
    Portanto, o cálculo do IQR deve ser aplicado na distribuição condicional de fraudes (ou com corte
    seguro) para NÃO eliminar a classe minoritária por engano.
    
    Args:
        X_train: Features de treino.
        y_train: Target de treino.
        features: Lista de features com maior correlação para aplicar o corte.
        factor: Multiplicador IQR conservador (2.5 a 3.0).
        
    Returns:
        (X_train_clean, y_train_clean)
    """
    df_train = X_train.copy()
    df_train["Class"] = y_train.values
    
    initial_len = len(df_train)
    initial_frauds = int(df_train["Class"].sum())
    
    indices_to_drop = set()
    
    # 1. Remover ruídos extremos da classe minoritária (fraudes aberrantes)
    for feat in features:
        fraud_series = df_train[df_train["Class"] == 1][feat]
        q25 = fraud_series.quantile(0.25)
        q75 = fraud_series.quantile(0.75)
        iqr = q75 - q25
        cutoff = iqr * factor
        lower = q25 - cutoff
        upper = q75 + cutoff
        
        fraud_outliers = fraud_series[(fraud_series < lower) | (fraud_series > upper)].index
        indices_to_drop.update(fraud_outliers)
        
    df_clean = df_train.drop(index=list(indices_to_drop))
    final_frauds = int(df_clean["Class"].sum())
    frauds_removed = initial_frauds - final_frauds
    
    print(f"[INFO] Tratamento Condicional de Outliers (fator {factor}):")
    print(f"       - Linhas totais de treino preservadas: {len(df_clean):,} de {initial_len:,}")
    print(f"       - Fraudes extremas removidas: {frauds_removed} | Fraudes mantidas: {final_frauds} (de {initial_frauds})")
    
    X_clean = df_clean.drop(columns=["Class"])
    y_clean = df_clean["Class"]
    
    return X_clean, y_clean
