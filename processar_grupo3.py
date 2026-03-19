from __future__ import annotations

import csv
import json
import re
import unicodedata
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

INPUT_ARFF = Path("mercadolimpo2.arff")
OUT_DIR = Path("saida_grupo3")

GROUP_CITIES = ["Belem", "Fortaleza", "Recife", "Curitiba"]
ALL_CITY_ATTRS = [
    "Belem",
    "Belo_Horizonte",
    "Curitiba",
    "Florianopolis",
    "Fortaleza",
    "Goiania",
    "Porto_Alegre",
    "Recife",
]

# Ordem mantida conforme o enunciado.
CATEGORY_ORDER = [
    "bebidas_alcoolicas",
    "refrigerantes_sucos",
    "queijos",
    "laticinios_sem_queijos",
    "chas",
    "cafes",
    "carnes_resfriadas_congeladas",
    "carnes_embutidas_defumadas",
    "peixes_frutosdomar",
    "frango",
    "frutas",
    "verduras_hortalicas",
    "tuberculos",
    "mercearia",
    "doces_industrializados",
    "enlatados_conserva_vacuo",
    "padaria",
    "massas",
    "pratos_prontos",
    "temperos_molhos",
    "salgados_salgadinhos",
    "matinais",
]

CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "bebidas_alcoolicas": [
        "cerveja",
        "vinho",
        "conhaque",
        "champanhe",
        "licor",
        "cidra",
        "vodka",
        "whisky",
        "rum",
        "gim",
        "tequila",
        "cachaca",
    ],
    "refrigerantes_sucos": [
        "refrigerante",
        "suco",
        "guarana",
        "aguadecoco",
        "concentrado",
        "polpa",
    ],
    "queijos": [
        "queijo",
        "requeijao",
    ],
    "laticinios_sem_queijos": [
        "leite",
        "iogurte",
        "manteiga",
        "creme",
        "yakult",
        "lactea",
        "coalhada",
        "nata",
    ],
    "chas": [
        "cha",
        "camomila",
        "ervamate",
    ],
    "cafes": [
        "cafe",
        "cappuccino",
    ],
    "carnes_resfriadas_congeladas": [
        "boi",
        "carnebovina",
        "carnedeboi",
        "alcatra",
        "costela",
        "fraldinha",
        "musculo",
        "filemignon",
        "figado",
        "tripa",
        "miudos",
        "medalhao",
        "peru",
    ],
    "carnes_embutidas_defumadas": [
        "salame",
        "salsicha",
        "linguica",
        "morcela",
        "presunto",
        "mortadela",
        "bacon",
        "paio",
    ],
    "peixes_frutosdomar": [
        "peixe",
        "sardinha",
        "atum",
        "bacalhau",
        "camarao",
        "lula",
        "polvo",
        "ameijoa",
        "molusco",
        "anchova",
    ],
    "frango": [
        "frango",
    ],
    "frutas": [
        "acai",
        "banana",
        "maca",
        "pera",
        "uva",
        "morango",
        "abacate",
        "caju",
        "coco",
        "abacaxi",
        "manga",
        "melancia",
        "melao",
        "papaya",
        "papaiada",
        "fruta",
    ],
    "verduras_hortalicas": [
        "alface",
        "repolho",
        "couve",
        "brocolis",
        "acelga",
        "chicoria",
        "aipo",
        "chuchu",
        "berinjela",
        "aspargo",
        "espinafre",
        "rucula",
        "horta",
        "verdura",
        "vegetais",
        "palmito",
    ],
    "tuberculos": [
        "batata",
        "mandioca",
        "cenoura",
        "inhame",
        "beterraba",
        "fecula",
        "araruta",
    ],
    "mercearia": [
        "arroz",
        "feijao",
        "farinha",
        "acucar",
        "sal",
        "canjica",
        "pipoca",
        "oleo",
        "trigo",
        "fermento",
        "graodebico",
        "graodeico",
        "amido",
    ],
    "doces_industrializados": [
        "doce",
        "chocolate",
        "bala",
        "chiclete",
        "trufa",
        "bombom",
        "sobremesa",
    ],
    "enlatados_conserva_vacuo": [
        "enlatad",
        "conserva",
        "vacuo",
    ],
    "padaria": [
        "pao",
        "paes",
        "bolo",
        "torta",
        "torrada",
        "padaria",
    ],
    "massas": [
        "massa",
        "macarrao",
        "espaguete",
        "lasanha",
        "nhoque",
        "canelone",
    ],
    "pratos_prontos": [
        "hamburguer",
        "almondega",
        "almondega",
        "nuggets",
        "croquete",
        "empanad",
        "espetinho",
        "pronto",
    ],
    "temperos_molhos": [
        "tempero",
        "molho",
        "pimenta",
        "canela",
        "cravo",
        "mostarda",
        "barbecue",
        "ajinomoto",
        "urucum",
        "louro",
    ],
    "salgados_salgadinhos": [
        "salgad",
        "chips",
        "snack",
        "aperitivo",
    ],
    "matinais": [
        "cereal",
        "bolacha",
        "biscoito",
        "nescau",
        "margarina",
        "granola",
        "matinal",
        "achocolat",
    ],
}


@dataclass
class ArffData:
    relation: str
    attributes: List[str]
    data: List[List[int]]


def normalize_text(text: str) -> str:
    n = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]", "", n.lower())


def parse_arff(path: Path) -> ArffData:
    relation = "dataset"
    attributes: List[str] = []
    data: List[List[int]] = []
    in_data = False

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("%"):
                continue
            low = line.lower()
            if low.startswith("@relation"):
                relation = line.split(maxsplit=1)[1] if " " in line else "dataset"
            elif low.startswith("@attribute"):
                m = re.match(r"@attribute\s+([^\s]+)\s+", line, flags=re.IGNORECASE)
                if m:
                    attributes.append(m.group(1))
            elif low == "@data":
                in_data = True
            elif in_data:
                row = [int(float(x.strip())) for x in line.split(",")]
                if len(row) == len(attributes):
                    data.append(row)

    return ArffData(relation=relation, attributes=attributes, data=data)


def to_dataframe(arff_data: ArffData) -> pd.DataFrame:
    return pd.DataFrame(arff_data.data, columns=arff_data.attributes)


def support_series(df: pd.DataFrame) -> pd.Series:
    return df.mean(axis=0)


def assign_categories(columns: List[str]) -> Tuple[Dict[str, List[str]], List[str]]:
    norm_cols = {c: normalize_text(c) for c in columns}
    assigned: Set[str] = set()
    cat_map: Dict[str, List[str]] = {cat: [] for cat in CATEGORY_ORDER}

    for cat in CATEGORY_ORDER:
        keys = CATEGORY_KEYWORDS[cat]
        for col in columns:
            if col in assigned:
                continue
            norm_col = norm_cols[col]
            if any(k in norm_col for k in keys):
                cat_map[cat].append(col)
                assigned.add(col)

    unassigned = [c for c in columns if c not in assigned]
    return cat_map, unassigned


def make_category_df(df_items: pd.DataFrame, cat_map: Dict[str, List[str]]) -> pd.DataFrame:
    out = pd.DataFrame(index=df_items.index)
    for cat in CATEGORY_ORDER:
        cols = cat_map[cat]
        if cols:
            out[cat] = (df_items[cols].sum(axis=1) > 0).astype(int)
        else:
            out[cat] = 0
    return out


def to_arff(path: Path, relation: str, df: pd.DataFrame) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(f"@relation {relation}\n\n")
        for col in df.columns:
            f.write(f"@attribute {col} numeric\n")
        f.write("\n@data\n")
        writer = csv.writer(f)
        for row in df.itertuples(index=False, name=None):
            writer.writerow(row)


def closed_itemsets(freq: pd.DataFrame) -> pd.DataFrame:
    rows = freq.sort_values("support", ascending=False).reset_index(drop=True)
    closed_flags = []
    itemsets = rows["itemsets"].tolist()
    supports = rows["support"].tolist()
    n = len(rows)

    for i in range(n):
        a = itemsets[i]
        sa = supports[i]
        is_closed = True
        for j in range(n):
            if i == j:
                continue
            b = itemsets[j]
            sb = supports[j]
            if a < b and abs(sa - sb) < 1e-12:
                is_closed = False
                break
        closed_flags.append(is_closed)

    return rows[pd.Series(closed_flags)].copy()


def rule_grid_search(df_final: pd.DataFrame) -> pd.DataFrame:
    all_rules: List[pd.DataFrame] = []
    minsup_values = [round(x, 2) for x in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]]
    minconf_values = [0.60, 0.70, 0.80, 0.90]

    for minsup, minconf in product(minsup_values, minconf_values):
        fi = apriori(
            df_final.astype(bool),
            min_support=minsup,
            use_colnames=True,
            max_len=4,
        )
        if fi.empty:
            continue
        rules = association_rules(fi, metric="confidence", min_threshold=minconf)
        if rules.empty:
            continue
        rules = rules.copy()
        rules["minsup_test"] = minsup
        rules["minconf_test"] = minconf
        all_rules.append(rules)

    if not all_rules:
        return pd.DataFrame()

    rules_all = pd.concat(all_rules, ignore_index=True)
    rules_all["antecedents"] = rules_all["antecedents"].apply(lambda s: ",".join(sorted(s)))
    rules_all["consequents"] = rules_all["consequents"].apply(lambda s: ",".join(sorted(s)))

    # Remove duplicadas do grid mantendo a de maior lift.
    rules_all = (
        rules_all.sort_values(["lift", "confidence", "support"], ascending=False)
        .drop_duplicates(subset=["antecedents", "consequents"])
        .reset_index(drop=True)
    )

    return rules_all


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    arff_data = parse_arff(INPUT_ARFF)
    df = to_dataframe(arff_data)

    # 1) Instancias do grupo.
    city_mask = df[GROUP_CITIES].sum(axis=1) > 0
    df_group = df.loc[city_mask].copy()

    # 2) Remove atributos de cidade.
    df_no_city = df_group.drop(columns=ALL_CITY_ATTRS)

    # 3) Remove atributos com suporte <= 2% (valor 1).
    supp_2 = support_series(df_no_city)
    cols_supp_gt_2 = supp_2[supp_2 > 0.02].index.tolist()
    removed_le_2 = sorted([c for c in df_no_city.columns if c not in cols_supp_gt_2])
    df_supp2 = df_no_city[cols_supp_gt_2].copy()

    # 4) Cria 22 categorias por agregacao.
    category_map, unassigned_items = assign_categories(df_supp2.columns.tolist())
    df_cat = make_category_df(df_supp2, category_map)

    # 5) Remove atributos originais (feito ao manter apenas categorias).
    # 6) Remove categorias com suporte < 10%.
    supp_cat = support_series(df_cat)
    final_cat_cols = supp_cat[supp_cat >= 0.10].index.tolist()
    removed_cat_lt_10 = sorted([c for c in df_cat.columns if c not in final_cat_cols])
    df_final = df_cat[final_cat_cols].copy()

    # 7) Apriori grid.
    rules_all = rule_grid_search(df_final)

    fi_for_closed = apriori(
        df_final.astype(bool), min_support=0.10, use_colnames=True, max_len=4
    )
    closed = closed_itemsets(fi_for_closed) if not fi_for_closed.empty else pd.DataFrame()

    # Ordenacao por lift e top 30.
    if not rules_all.empty:
        rules_sorted = rules_all.sort_values(["lift", "confidence", "support"], ascending=False)
        top30 = rules_sorted.head(30).copy()
    else:
        rules_sorted = pd.DataFrame()
        top30 = pd.DataFrame()

    # Saidas.
    df_final.to_csv(OUT_DIR / "base_grupo3_final.csv", index=False)
    to_arff(OUT_DIR / "base_grupo3_final.arff", "mercado_grupo3_categorizado", df_final)

    if not rules_sorted.empty:
        cols = [
            "antecedents",
            "consequents",
            "support",
            "confidence",
            "lift",
            "leverage",
            "conviction",
            "minsup_test",
            "minconf_test",
        ]
        rules_sorted[cols].to_csv(OUT_DIR / "regras_apriori_todas.csv", index=False)
        top30[cols].to_csv(OUT_DIR / "regras_apriori_top30_lift.csv", index=False)

    if not closed.empty:
        closed_out = closed.copy()
        closed_out["itemsets"] = closed_out["itemsets"].apply(lambda s: ",".join(sorted(s)))
        closed_out.to_csv(OUT_DIR / "itemsets_fechados_minsup10.csv", index=False)

    cat_map_json = {k: v for k, v in category_map.items()}

    summary = {
        "instancias_totais": int(df.shape[0]),
        "instancias_grupo3": int(df_group.shape[0]),
        "atributos_originais": int(df.shape[1]),
        "atributos_sem_cidades": int(df_no_city.shape[1]),
        "atributos_removidos_suporte_le_2": len(removed_le_2),
        "atributos_pos_suporte_2": int(df_supp2.shape[1]),
        "categorias_total": len(CATEGORY_ORDER),
        "categorias_final_suporte_ge_10": int(df_final.shape[1]),
        "categorias_removidas_suporte_lt_10": removed_cat_lt_10,
        "nao_classificados_total": len(unassigned_items),
        "nao_classificados": unassigned_items,
        "regras_total_unicas_grid": int(rules_all.shape[0]) if not rules_all.empty else 0,
        "regras_top30_geradas": int(top30.shape[0]) if not top30.empty else 0,
    }

    (OUT_DIR / "mapeamento_categorias.json").write_text(
        json.dumps(cat_map_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT_DIR / "atributos_removidos_suporte_le_2.txt").write_text(
        "\n".join(removed_le_2), encoding="utf-8"
    )
    (OUT_DIR / "atributos_nao_classificados.txt").write_text(
        "\n".join(unassigned_items), encoding="utf-8"
    )
    (OUT_DIR / "resumo_execucao.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
