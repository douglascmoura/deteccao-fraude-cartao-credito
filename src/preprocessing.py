"""
Módulo de pré-processamento de dados para detecção de fraudes.
Contém funções para carregamento seguro, escalonamento robusto e particionamento estratificado.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Carrega o dataset de transações de cartão de crédito.
    
    Args:
        filepath: Caminho para o arquivo CSV. Se None, busca em caminhos padrão relativos.
        
    Returns:
        pd.DataFrame com os dados carregados.
    """
    if filepath is None:
        possible_paths = [
            "../data/creditcard.csv",
            "data/creditcard.csv",
            "../../credit_card_fraud_detection/data/creditcard.csv"
        ]
        for p in possible_paths:
            if os.path.exists(p):
                filepath = p
                break
        if filepath is None:
            raise FileNotFoundError("Arquivo 'creditcard.csv' não encontrado nos caminhos padrão.")

    print(f"[INFO] Carregando dados de: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[INFO] Dataset carregado com sucesso: {df.shape[0]:,} linhas e {df.shape[1]} colunas.")
    return df


def prepare_and_scale_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica RobustScaler nas variáveis 'Time' e 'Amount', mantendo as componentes V1-V28 intactas.
    RobustScaler é ideal pois utiliza mediana e intervalo interquartil (IQR), sendo resistente a outliers.
    
    Args:
        df: DataFrame original contendo 'Time', 'Amount' e 'V1' a 'V28'.
        
    Returns:
        pd.DataFrame com 'scaled_time' e 'scaled_amount' e as colunas originais removidas.
    """
    df_processed = df.copy()
    scaler = RobustScaler()
    
    df_processed["scaled_amount"] = scaler.fit_transform(df_processed[["Amount"]])
    df_processed["scaled_time"] = scaler.fit_transform(df_processed[["Time"]])
    
    scaled_amount = df_processed["scaled_amount"]
    scaled_time = df_processed["scaled_time"]
    
    df_processed.drop(columns=["Time", "Amount", "scaled_amount", "scaled_time"], inplace=True)
    df_processed.insert(0, "scaled_time", scaled_time)
    df_processed.insert(1, "scaled_amount", scaled_amount)
    
    return df_processed


def split_data(
    df: pd.DataFrame,
    target_col: str = "Class",
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Realiza divisão estratificada entre treino e teste para dados massivamente desbalanceados.
    Previne rigorosamente qualquer vazamento de dados (data leakage).
    
    Args:
        df: DataFrame com features e variável alvo.
        target_col: Nome da coluna alvo (default: 'Class').
        test_size: Proporção do conjunto de teste (default: 0.20).
        random_state: Semente de reprodutibilidade.
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    fraud_train_pct = (y_train.sum() / len(y_train)) * 100
    fraud_test_pct = (y_test.sum() / len(y_test)) * 100
    
    print(f"[INFO] Treino: {len(y_train):,} amostras | Fraudes: {y_train.sum():,} ({fraud_train_pct:.3f}%)")
    print(f"[INFO] Teste:  {len(y_test):,} amostras | Fraudes: {y_test.sum():,} ({fraud_test_pct:.3f}%)")
    
    return X_train, X_test, y_train, y_test
