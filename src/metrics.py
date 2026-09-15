"""
Módulo para avaliação e cálculo de métricas em cenários de fraude desbalanceada.
Contém funções estatísticas (PR-AUC, Recall, Precision) e dashboards visuais interativos em Plotly Dark.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve
)
from typing import Dict, Tuple, Optional, Any, List


def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calcula o conjunto completo de métricas adequadas para dados massivamente desbalanceados.
    
    Args:
        y_true: Classes reais (0 ou 1).
        y_pred: Classes preditas (0 ou 1).
        y_prob: Probabilidades preditas da classe 1 (opcional para AUCs).
        
    Returns:
        Dicionário com Precision, Recall, F1, PR-AUC e ROC-AUC.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "True Positives (TP)": int(tp),
        "False Positives (FP)": int(fp),
        "False Negatives (FN)": int(fn),
        "True Negatives (TN)": int(tn),
    }
    
    if y_prob is not None:
        metrics["PR-AUC"] = average_precision_score(y_true, y_prob)
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
        
    return metrics


def plot_confusion_matrix_plotly(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Matriz de Confusão",
    height: int = 460,
    width: int = 680
) -> go.Figure:
    """
    Renderiza uma Matriz de Confusão interativa no Plotly Dark (#0B1320).
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    z_data = cm[::-1]
    pct_tn = tn / (tn + fp) * 100
    pct_fp = fp / (tn + fp) * 100
    pct_fn = fn / (fn + tp) * 100
    pct_tp = tp / (fn + tp) * 100
    
    text_labels = [
        [f"FN (Fraude Perdida)<br><b>{fn:,}</b> ({pct_fn:.1f}%)", f"TP (Fraude Interceptada)<br><b>{tp:,}</b> ({pct_tp:.1f}%)"],
        [f"TN (Legítima Correta)<br><b>{tn:,}</b> ({pct_tn:.1f}%)", f"FP (Alarme Falso)<br><b>{fp:,}</b> ({pct_fp:.1f}%)"]
    ]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=['Previsto Legítimo (0)', 'Previsto Fraude (1)'],
        y=['Real Fraude (1)', 'Real Legítimo (0)'],
        colorscale=[[0.0, '#182231'], [0.5, '#1E293B'], [1.0, '#E2E8F0']],
        showscale=False,
        text=text_labels,
        texttemplate="%{text}",
        textfont=dict(size=13, color="#445263", family="Arial"),
        hovertemplate="<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Transações:</b> %{z:,}<extra></extra>"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18, family="Arial", color="#E2E8F0")),
        xaxis_title="Classe Predita pelo Modelo",
        yaxis_title="Classe Real da Transação",
        height=height,
        width=width,
        plot_bgcolor="#0B1320",
        paper_bgcolor='#0B1320',
        font=dict(family="Arial", color="#E2E8F0"),
        margin=dict(t=70, b=50, l=80, r=40)
    )
    
    fig.update_xaxes(showgrid=False, color="#E2E8F0")
    fig.update_yaxes(showgrid=False, color="#E2E8F0")
    
    return fig


def plot_model_performance_dashboard(
    models: Dict[str, Any],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    height: int = 1050,
    width: int = 1200
) -> go.Figure:
    """
    Cria um Dashboard 3x3 interativo em Plotly Dark comparando múltiplos modelos:
    - Coluna 1: Matriz de Confusão (Heatmap)
    - Coluna 2: Curva ROC (com área sombreada dourada)
    - Coluna 3: Curva Precision-Recall (com área sombreada dourada)
    
    Args:
        models: Dicionário {'Nome': modelo_treinado}
        X_val: Features do teste
        y_val: Target real do teste
    """
    subplot_titles = []
    for name in models.keys():
        subplot_titles.extend([f"{name} - Matriz", f"{name} - ROC", f"{name} - PR"])
        
    fig = make_subplots(
        rows=len(models),
        cols=3,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.08,
        vertical_spacing=0.09
    )
    
    y_val_arr = np.array(y_val)
    no_skill_ratio = y_val_arr.sum() / len(y_val_arr)
    
    for row_idx, (name, model) in enumerate(models.items(), start=1):
        y_pred = model.predict(X_val)
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            scores = model.decision_function(X_val)
            y_prob = (scores - scores.min()) / (scores.max() - scores.min())
        else:
            y_prob = y_pred.astype(float)
            
        # 1. Matriz de Confusão (Coluna 1)
        cm = confusion_matrix(y_val_arr, y_pred)
        z_data = cm[::-1]
        
        heatmap = go.Heatmap(
            z=z_data,
            x=['Legítimo (0)', 'Fraude (1)'],
            y=['Fraude (1)', 'Legítimo (0)'],
            colorscale=[[0.0, '#182231'], [0.5, '#1E293B'], [1.0, '#E2E8F0']],
            showscale=False,
            text=z_data,
            texttemplate="<b>%{text:,}</b>",
            textfont=dict(size=14, color="#445263", family="Arial"),
            hovertemplate="<b>Real:</b> %{y}<br><b>Predito:</b> %{x}<br><b>Qtd:</b> %{z:,}<extra></extra>"
        )
        fig.add_trace(heatmap, row=row_idx, col=1)
        
        # 2. Curva ROC (Coluna 2)
        fpr, tpr, _ = roc_curve(y_val_arr, y_prob)
        roc_auc = roc_auc_score(y_val_arr, y_prob)
        
        fig.add_trace(
            go.Scatter(
                x=fpr, y=tpr,
                name=f'ROC {name}',
                fill='tozeroy',
                fillcolor='rgba(226, 232, 240, 0.12)',
                line=dict(color='#E2E8F0', width=2.5),
                hovertemplate=f"<b>{name}</b><br>FPR: %{{x:.3f}}<br>TPR (Recall): %{{y:.3f}}<extra></extra>"
            ),
            row=row_idx, col=2
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[0, 1],
                line=dict(dash='dash', color='#5E7E99', width=1.2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row_idx, col=2
        )
        
        # 3. Curva Precision-Recall (Coluna 3)
        precision, recall, _ = precision_recall_curve(y_val_arr, y_prob)
        pr_auc = average_precision_score(y_val_arr, y_prob)
        
        fig.add_trace(
            go.Scatter(
                x=recall, y=precision,
                name=f'PR {name}',
                fill='tozeroy',
                fillcolor='rgba(229, 62, 62, 0.20)',
                line=dict(color='#E53E3E', width=2.5),
                hovertemplate=f"<b>{name}</b><br>Recall: %{{x:.3f}}<br>Precision: %{{y:.3f}}<extra></extra>"
            ),
            row=row_idx, col=3
        )
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[no_skill_ratio, no_skill_ratio],
                line=dict(dash='dash', color='#5E7E99', width=1.2),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=row_idx, col=3
        )
        
    fig.update_layout(
        height=height,
        width=width,
        showlegend=False,
        separators=",.",
        title=dict(
            text="PAINEL COMPARATIVO DE PERFORMANCE DOS MODELOS",
            x=0.5,
            font=dict(size=22, family="Arial", color="#E2E8F0")
        ),
        plot_bgcolor='#0B1320',
        paper_bgcolor='#0B1320',
        font=dict(family="Arial", color="#E2E8F0"),
        margin=dict(t=90, b=50, l=70, r=40)
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color='#E2E8F0', family="Arial")
        
    for i in range(1, len(models) + 1):
        fig.update_xaxes(title_text="Predito", row=i, col=1, title_font=dict(size=11), showgrid=False, color="#E2E8F0")
        fig.update_yaxes(title_text="Real", row=i, col=1, title_font=dict(size=11), showgrid=False, color="#E2E8F0")
        fig.update_xaxes(title_text="FPR", range=[0, 1], row=i, col=2, showgrid=True, gridcolor="#1E293B", color="#E2E8F0")
        fig.update_yaxes(title_text="TPR (Recall)", range=[0, 1.05], row=i, col=2, showgrid=True, gridcolor="#1E293B", color="#E2E8F0")
        fig.update_xaxes(title_text="Recall", range=[0, 1], row=i, col=3, showgrid=True, gridcolor="#1E293B", color="#E2E8F0")
        fig.update_yaxes(title_text="Precision", range=[0, 1.05], row=i, col=3, showgrid=True, gridcolor="#1E293B", color="#E2E8F0")
        
    return fig


def plot_feature_importances(
    model: Any,
    feature_names: List[str],
    top_n: int = 15,
    title: Optional[str] = None
) -> go.Figure:
    """
    Gera um gráfico interativo de barras horizontais no Plotly Dark
    exibindo a importância das variáveis do modelo campeão (árvores/boosting).
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("O modelo fornecido não possui o atributo 'feature_importances_'.")
        
    importances = model.feature_importances_
    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=True)
    
    top_df = feat_df.tail(top_n)
    
    if title is None:
        title = f"Top {top_n} Variáveis Mais Determinantes na Detecção de Fraude"
        
    fig = px.bar(
        top_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title=title,
        color="Importance",
        color_continuous_scale=[[0.0, '#1E293B'], [0.5, '#E2E8F0'], [1.0, '#E53E3E']]
    )
    
    fig.update_traces(
        textposition="outside",
        texttemplate="%{x:.4f}",
        textfont=dict(color="#E2E8F0", size=12),
        hovertemplate="<b>Variável:</b> %{y}<br><b>Peso de Importância:</b> %{x:.4f}<extra></extra>"
    )
    
    fig.update_layout(
        separators=",.",
        title=dict(x=0.5, font=dict(size=20, family="Arial", color="#E2E8F0")),
        font=dict(family="Arial", size=13, color="#E2E8F0"),
        plot_bgcolor='#0B1320',
        paper_bgcolor='#0B1320',
        xaxis_title="Peso Relativo de Importância (Gini / Ganho)",
        yaxis_title="",
        coloraxis_showscale=False,
        margin=dict(t=80, b=50, l=90, r=60),
        xaxis=dict(showgrid=True, gridcolor="#1E293B", color="#E2E8F0"),
        yaxis=dict(showgrid=False, color="#E2E8F0", dtick=1)
    )
    
    return fig


def plot_curves_comparison(
    models_dict: Dict[str, Tuple[np.ndarray, np.ndarray]],
    figsize: Tuple[int, int] = (15, 6)
) -> None:
    """
    Plota lado a lado a Curva Precision-Recall (PR-AUC) e a Curva ROC-AUC para múltiplos modelos em Matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    for name, (y_true, y_prob) in models_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
        ax1.plot(recall, precision, lw=2, label=f"{name} (PR-AUC = {pr_auc:.3f})")
        
    no_skill = y_true.sum() / len(y_true)
    ax1.axhline(no_skill, color="gray", linestyle="--", label=f"Baseline Aleatório ({no_skill:.4f})")
    ax1.set_title("Curva Precision-Recall (Foco na Classe Positiva)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Recall (Sensibilidade)", fontweight="bold")
    ax1.set_ylabel("Precision (Precisão)", fontweight="bold")
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.05, 1.05])
    ax1.legend(loc="lower left")
    ax1.grid(alpha=0.3)
    
    for name, (y_true, y_prob) in models_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = roc_auc_score(y_true, y_prob)
        ax2.plot(fpr, tpr, lw=2, label=f"{name} (ROC-AUC = {roc_auc:.3f})")
        
    ax2.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Aleatório (0.500)")
    ax2.set_title("Curva ROC-AUC", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Taxa de Falsos Positivos (FPR)", fontweight="bold")
    ax2.set_ylabel("Taxa de Verdadeiros Positivos (TPR)", fontweight="bold")
    ax2.set_ylim([-0.05, 1.05])
    ax2.set_xlim([-0.05, 1.05])
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
