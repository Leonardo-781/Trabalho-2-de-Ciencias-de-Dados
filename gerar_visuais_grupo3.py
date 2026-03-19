from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from textwrap import fill

BASE_DIR = Path("saida_grupo3")
BASE_FILE = BASE_DIR / "base_grupo3_final.csv"
RULES_FILE = BASE_DIR / "regras_apriori_top30_lift.csv"

OUT_SUPPORT = BASE_DIR / "grafico_suporte_categorias.png"
OUT_TOP_RULES = BASE_DIR / "grafico_top10_regras_lift.png"
OUT_SCATTER = BASE_DIR / "grafico_scatter_regras.png"
OUT_HEATMAP = BASE_DIR / "grafico_heatmap_correlacao.png"


def setup_style() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["font.size"] = 10


def label_pt(col: str) -> str:
    return col.replace("_", " ").title()


def shorten_rule(text: str, width: int = 52) -> str:
    return fill(text, width=width)


def plot_support(df: pd.DataFrame) -> None:
    support_pct = (df.mean() * 100).sort_values(ascending=False).head(12)
    y_labels = [label_pt(c) for c in support_pct.index]

    plt.figure(figsize=(10, 6.5))
    ax = sns.barplot(x=support_pct.values, y=y_labels, color="#2A9D8F")
    ax.set_title("Top 12 Categorias por Suporte (%) - Grupo 3", fontsize=13)
    ax.set_xlabel("Suporte (%)")
    ax.set_ylabel("Categoria")
    ax.set_xlim(0, 100)

    for i, v in enumerate(support_pct.values):
        ax.text(v + 0.7, i, f"{v:.1f}%", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT_SUPPORT)
    plt.close()


def plot_top_rules(rules: pd.DataFrame) -> None:
    top10 = rules.nlargest(10, "lift").copy().reset_index(drop=True)
    top10["regra"] = (
        top10["antecedents"].astype(str) + " -> " + top10["consequents"].astype(str)
    )
    top10["regra"] = top10["regra"].apply(shorten_rule)

    plt.figure(figsize=(11, 8))
    ax = sns.barplot(x="lift", y="regra", data=top10, color="#E76F51")
    ax.set_title("Top 10 Regras por Lift", fontsize=13)
    ax.set_xlabel("Lift")
    ax.set_ylabel("Regra (Antecedente -> Consequente)")

    for i, row in top10.iterrows():
        ax.text(row["lift"] + 0.01, i, f"lift={row['lift']:.2f} | conf={row['confidence']:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT_TOP_RULES)
    plt.close()


def plot_scatter_rules(rules: pd.DataFrame) -> None:
    top30 = rules.nlargest(30, "lift").copy().reset_index(drop=True)

    plt.figure(figsize=(9.5, 6.5))
    ax = sns.scatterplot(
        data=top30,
        x="confidence",
        y="lift",
        size="support",
        hue="support",
        palette="viridis",
        sizes=(80, 420),
        alpha=0.8,
    )
    ax.set_title("Top 30 Regras: Confiança x Lift (tamanho/cor = Suporte)", fontsize=13)
    ax.set_xlabel("Confiança")
    ax.set_ylabel("Lift")
    ax.legend(title="Suporte", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER)
    plt.close()


def plot_heatmap(df: pd.DataFrame) -> None:
    corr = df.corr()
    # Seleciona 10 categorias com maior suporte para reduzir poluicao visual.
    top_cols = df.mean().sort_values(ascending=False).head(10).index.tolist()
    corr_top = corr.loc[top_cols, top_cols]

    plt.figure(figsize=(8.5, 7.5))
    ax = sns.heatmap(
        corr_top,
        cmap="YlGnBu",
        center=0,
        square=True,
        annot=True,
        fmt=".2f",
        cbar_kws={"label": "Correlacao"},
    )
    ax.set_title("Correlacao (Top 10 categorias por suporte)", fontsize=13)
    ax.set_xticklabels([label_pt(c) for c in top_cols], rotation=45, ha="right")
    ax.set_yticklabels([label_pt(c) for c in top_cols], rotation=0)
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP)
    plt.close()


def main() -> None:
    setup_style()

    df = pd.read_csv(BASE_FILE)
    rules = pd.read_csv(RULES_FILE)

    plot_support(df)
    plot_top_rules(rules)
    plot_scatter_rules(rules)
    plot_heatmap(df)

    print("Graficos gerados:")
    print(f"- {OUT_SUPPORT}")
    print(f"- {OUT_TOP_RULES}")
    print(f"- {OUT_SCATTER}")
    print(f"- {OUT_HEATMAP}")


if __name__ == "__main__":
    main()
