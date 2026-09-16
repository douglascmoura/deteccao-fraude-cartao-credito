"""
Módulo de inteligência de negócios e impacto financeiro.
Traduz métricas probabilísticas de Machine Learning em dólares e ROI real de prevenção a fraudes.
Inclui otimização matemática de limiares (Threshold Tuning) e gráficos interativos no Plotly Dark.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Tuple, Any, Optional, List
from sklearn.metrics import precision_score, recall_score


def calculate_financial_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = 15.0
) -> Dict[str, Any]:
    """
    Calcula o impacto financeiro de um modelo em dólares com base nos valores reais das transações.
    
    Args:
        y_true: Classes reais (0 ou 1).
        y_pred: Classes preditas (0 ou 1).
        amounts: Valores originais das transações ($) correspondentes.
        cost_fp: Custo operacional fixo de verificação manual ou atrito por falso positivo ($).
        
    Returns:
        Dicionário com prejuízos, custos operacionais e economias geradas.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    amounts = np.array(amounts)
    
    fn_mask = (y_true == 1) & (y_pred == 0)
    fp_mask = (y_true == 0) & (y_pred == 1)
    tp_mask = (y_true == 1) & (y_pred == 1)
    
    total_fraud_attempted = float(amounts[y_true == 1].sum())
    fn_losses = float(amounts[fn_mask].sum())
    tp_saved = float(amounts[tp_mask].sum())
    fp_count = int(fp_mask.sum())
    fp_cost_total = float(fp_count * cost_fp)
    
    total_financial_cost = fn_losses + fp_cost_total
    
    return {
        "Total Fraude Tentada ($)": total_fraud_attempted,
        "Fraude Prevenida ($)": tp_saved,
        "Prejuízo com Fraudes Não Detectadas - FN ($)": fn_losses,
        "Qtd Alarmes Falsos (FP)": fp_count,
        "Custo Operacional de Alarmes Falsos - FP ($)": fp_cost_total,
        "Custo Total Operacional ($)": total_financial_cost,
        "Taxa de Recuperação Financeira (%)": (tp_saved / total_fraud_attempted * 100) if total_fraud_attempted > 0 else 0
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    cost_fp: float = 15.0,
    thresholds: np.ndarray = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Analisa uma grade de limiares de decisão (thresholds) e encontra o ponto ótimo que
    minimiza o custo financeiro total (Perda por FN + Custo por FP).
    
    Returns:
        df_curves: Histórico financeiro para cada limiar.
        best_decision: Detalhes do melhor limiar financeiro encontrado.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)
        
    records = []
    
    for th in thresholds:
        y_pred = (y_prob >= th).astype(int)
        impact = calculate_financial_impact(y_true, y_pred, amounts, cost_fp=cost_fp)
        
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        
        records.append({
            "threshold": th,
            "recall": rec,
            "precision": prec,
            "fn_losses": impact["Prejuízo com Fraudes Não Detectadas - FN ($)"],
            "fp_costs": impact["Custo Operacional de Alarmes Falsos - FP ($)"],
            "total_cost": impact["Custo Total Operacional ($)"],
            "saved_amount": impact["Fraude Prevenida ($)"]
        })
        
    df_curves = pd.DataFrame(records)
    
    best_idx = df_curves["total_cost"].idxmin()
    best_row = df_curves.loc[best_idx]
    
    cost_no_model = float(amounts[y_true == 1].sum())
    savings_vs_no_model = cost_no_model - best_row["total_cost"]
    
    best_decision = {
        "best_threshold": best_row["threshold"],
        "min_cost": best_row["total_cost"],
        "cost_no_model": cost_no_model,
        "savings_vs_no_model": savings_vs_no_model,
        "roi_improvement_pct": (savings_vs_no_model / cost_no_model * 100) if cost_no_model > 0 else 0,
        "recall_at_best": best_row["recall"],
        "precision_at_best": best_row["precision"]
    }
    
    return df_curves, best_decision


def plot_threshold_curves_plotly(
    df_curves: pd.DataFrame,
    best_decision: Dict[str, Any],
    height: int = 520,
    width: int = 1300
) -> go.Figure:
    """
    Renderiza um painel interativo no Plotly Dark (#0B1320) com 2 subplots:
    - Subplot 1: Curvas de Custo Financeiro ($) vs. Limiar com destaque no ponto ótimo
    - Subplot 2: Trade-off de Métricas (Recall vs. Precisão) vs. Limiar
    """
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Otimização Financeira: Custo Total vs. Limiar de Decisão",
            "Trade-off Estatístico: Recall vs. Precisão"
        ),
        horizontal_spacing=0.10
    )
    
    best_th = best_decision["best_threshold"]
    min_cost = best_decision["min_cost"]
    
    # 1. Curvas Financeiras (Subplot 1)
    fig.add_trace(
        go.Scatter(
            x=df_curves["threshold"],
            y=df_curves["total_cost"],
            name="Custo Total Operacional",
            line=dict(color="#E2E8F0", width=3.0),
            hovertemplate="Custo Total: <b>$%{y:,.2f}</b><extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_curves["threshold"],
            y=df_curves["fn_losses"],
            name="Prejuízo Fraudes Perdidas (FN)",
            line=dict(color="#E53E3E", width=2.0, dash="dash"),
            hovertemplate="Prejuízo FN: <b>$%{y:,.2f}</b><extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_curves["threshold"],
            y=df_curves["fp_costs"],
            name="Custo Alarmes Falsos (FP)",
            line=dict(color="#94A3B8", width=1.8, dash="dot"),
            hovertemplate="Custo FP: <b>$%{y:,.2f}</b><extra></extra>"
        ),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=[best_th],
            y=[min_cost],
            name="Limiar Ótimo",
            mode="markers",
            marker=dict(color="#E53E3E", size=11, symbol="diamond", line=dict(color="#E2E8F0", width=2.0)),
            hovertemplate=f"Ponto Ótimo: <b>${min_cost:,.2f}</b><extra></extra>"
        ),
        row=1, col=1
    )
    
    # 2. Trade-off Recall vs Precisão (Subplot 2)
    fig.add_trace(
        go.Scatter(
            x=df_curves["threshold"],
            y=df_curves["recall"],
            name="Recall (Sensibilidade)",
            line=dict(color="#10B981", width=2.5),
            hovertemplate="Recall: <b>%{y:.2%}</b><extra></extra>"
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=df_curves["threshold"],
            y=df_curves["precision"],
            name="Precisão",
            line=dict(color="#38BDF8", width=2.5),
            hovertemplate="Precisão: <b>%{y:.2%}</b><extra></extra>"
        ),
        row=1, col=2
    )
    
    for col_idx in [1, 2]:
        fig.add_vline(
            x=best_th,
            line_dash="dash",
            line_color="#E2E8F0",
            line_width=1.5,
            row=1, col=col_idx
        )
        
    fig.update_layout(
        height=height,
        width=width,
        separators=",.",
        title=dict(
            text="ANÁLISE DE THRESHOLD TUNING & IMPACTO FINANCEIRO",
            x=0.5,
            font=dict(size=20, family="Arial", color="#E2E8F0")
        ),
        plot_bgcolor='#0B1320',
        paper_bgcolor='#0B1320',
        font=dict(family="Arial", color="#E2E8F0"),
        margin=dict(t=80, b=60, l=70, r=40),
        hovermode="x unified",
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color="#E2E8F0",
            font_family="Arial",
            font_size=12,
            bordercolor="#445263"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.28,
            xanchor="center",
            x=0.5,
            font=dict(size=12, color="#E2E8F0")
        )
    )
    
    for annotation in fig['layout']['annotations']:
        annotation['font'] = dict(size=13, color='#E2E8F0', family="Arial")
        
    fig.update_xaxes(
        title_text="Limiar de Decisão (Threshold)", 
        showgrid=True, gridcolor="#1E293B", color="#E2E8F0",
        showspikes=True, spikecolor="#E2E8F0", spikethickness=1, spikemode="across",
        row=1, col=1
    )
    fig.update_yaxes(title_text="Custo Operacional ($ USD)", showgrid=True, gridcolor="#1E293B", color="#E2E8F0", row=1, col=1)
    
    fig.update_xaxes(
        title_text="Limiar de Decisão (Threshold)", 
        showgrid=True, gridcolor="#1E293B", color="#E2E8F0",
        showspikes=True, spikecolor="#E2E8F0", spikethickness=1, spikemode="across",
        row=1, col=2
    )
    fig.update_yaxes(title_text="Score (0 a 1.0)", range=[-0.05, 1.05], showgrid=True, gridcolor="#1E293B", color="#E2E8F0", row=1, col=2)
    
    return fig


def plot_threshold_curves(
    df_curves: pd.DataFrame,
    best_decision: Dict[str, Any],
    figsize: Tuple[int, int] = (16, 6)
) -> None:
    """
    Plota as curvas de Custo Financeiro e Trade-off de Métricas em Matplotlib.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    best_th = best_decision["best_threshold"]
    
    ax1.plot(df_curves["threshold"], df_curves["total_cost"], label="Custo Total Operacional", color="#D32F2F", lw=2.5)
    ax1.plot(df_curves["threshold"], df_curves["fn_losses"], label="Perda por Fraudes Não Detectadas (FN)", color="#FF9800", linestyle="--")
    ax1.plot(df_curves["threshold"], df_curves["fp_costs"], label="Custo Operacional de Falsos Positivos (FP)", color="#1976D2", linestyle=":")
    
    ax1.axvline(best_th, color="black", linestyle="--", alpha=0.8, label=f"Limiar Ótimo = {best_th:.2f}")
    ax1.scatter([best_th], [best_decision["min_cost"]], color="#D32F2F", s=100, zorder=5)
    
    ax1.set_title("Otimização Financeira: Custo vs Limiar de Decisão", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Limiar de Decisão (Threshold)", fontweight="bold")
    ax1.set_ylabel("Custo Operacional ($ USD)", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    ax2.plot(df_curves["threshold"], df_curves["recall"], label="Recall (Sensibilidade)", color="#388E3C", lw=2)
    ax2.plot(df_curves["threshold"], df_curves["precision"], label="Precisão", color="#7B1FA2", lw=2)
    ax2.axvline(best_th, color="black", linestyle="--", alpha=0.8, label=f"Limiar Ótimo = {best_th:.2f}")
    
    ax2.set_title("Trade-off de Métricas: Recall vs Precisão", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Limiar de Decisão (Threshold)", fontweight="bold")
    ax2.set_ylabel("Score (0 a 1)", fontweight="bold")
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def simulate_fp_cost_sensitivity(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    fp_costs: Optional[List[float]] = None
) -> pd.DataFrame:
    """
    Avalia a sensibilidade do modelo financeiro sob diferentes cenários de custo operacional por falso positivo.
    """
    if fp_costs is None:
        fp_costs = [5.0, 10.0, 15.0, 30.0, 50.0]
        
    records = []
    for cost in fp_costs:
        _, decision = optimize_threshold(y_true, y_prob, amounts, cost_fp=cost)
        records.append({
            "Custo FP Unitário": f"$ {cost:,.2f}".replace('.', ','),
            "Limiar Ótimo (Threshold)": f"{decision['best_threshold']:.2f}".replace('.', ','),
            "Recall Transações (%)": f"{decision['recall_at_best']:.2%}",
            "Precisão (%)": f"{decision['precision_at_best']:.2%}",
            "Custo Total da Operação": f"$ {decision['min_cost']:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.'),
            "Economia Líquida Gerada": f"$ {decision['savings_vs_no_model']:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.'),
            "Redução do Prejuízo (%)": f"{decision['roi_improvement_pct']:.1f}%".replace('.', ',')
        })
        
    return pd.DataFrame(records)


def plot_tp_vs_fn_amounts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray,
    height: int = 460,
    width: int = 1050
) -> go.Figure:
    """
    Plota a distribuição comparativa dos valores monetários ($) das fraudes capturadas (TP)
    versus fraudes não detectadas (FN) no Plotly Dark (#0B1320).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    amounts = np.array(amounts)
    
    tp_amounts = amounts[(y_true == 1) & (y_pred == 1)]
    fn_amounts = amounts[(y_true == 1) & (y_pred == 0)]
    
    df_plot = pd.DataFrame({
        "Amount": np.concatenate([tp_amounts, fn_amounts]),
        "Grupo": ["Fraudes Capturadas (TP)"] * len(tp_amounts) + ["Fraudes Não Detectadas (FN)"] * len(fn_amounts)
    })
    
    fig = px.box(
        df_plot,
        x="Grupo",
        y="Amount",
        color="Grupo",
        color_discrete_map={
            "Fraudes Capturadas (TP)": "#E2E8F0",
            "Fraudes Não Detectadas (FN)": "#E53E3E"
        },
        points="all",
        title="Distribuição de Valores ($): Fraudes Capturadas (TP) vs. Perdidas (FN)"
    )
    
    fig.update_traces(
        marker=dict(size=6, opacity=0.85, line=dict(width=1, color="#E2E8F0")),
        boxmean=True
    )
    
    fig.update_layout(
        height=height,
        width=width,
        separators=",.",
        title=dict(x=0.5, font=dict(size=18, family="Arial", color="#E2E8F0")),
        xaxis_title="",
        yaxis_title="Valor da Transação ($ - Escala Logarítmica)",
        font=dict(family="Arial", size=13, color="#E2E8F0"),
        plot_bgcolor='#0B1320',
        paper_bgcolor='#0B1320',
        showlegend=False,
        yaxis=dict(type="log", showgrid=True, gridcolor="#1E293B", color="#E2E8F0"),
        xaxis=dict(showgrid=False, color="#E2E8F0")
    )
    
    return fig

