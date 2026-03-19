# Trabalho 2 de Ciencias de Dados

Projeto de mineracao de regras de associacao (Grupo 3) com foco em:
- filtragem de instancias por cidades do grupo,
- reducao de dimensionalidade por suporte,
- agregacao de itens em categorias de negocio,
- geracao de regras com Apriori,
- visualizacao dos principais resultados.

Repositorio: https://github.com/Leonardo-781/Trabalho-2-de-Ciencias-de-Dados

## Objetivo
Extrair padroes de compra relevantes para o Grupo 3 e produzir uma base final interpretavel, regras de associacao e graficos para analise.

## Base de dados
- Arquivo de entrada: `mercadolimpo2.arff`
- Instancias totais: 1540
- Instancias do Grupo 3: 757
- Atributos originais: 741

## Estrutura do repositorio
```text
.
├─ mercadolimpo2.arff
├─ processar_grupo3.py
├─ gerar_visuais_grupo3.py
└─ saida_grupo3/
   ├─ base_grupo3_final.csv
   ├─ base_grupo3_final.arff
   ├─ regras_apriori_todas.csv
   ├─ regras_apriori_top30_lift.csv
   ├─ itemsets_fechados_minsup10.csv
   ├─ mapeamento_categorias.json
   ├─ resumo_execucao.json
   ├─ atributos_removidos_suporte_le_2.txt
   ├─ atributos_nao_classificados.txt
   ├─ grafico_suporte_categorias.png
   ├─ grafico_top10_regras_lift.png
   ├─ grafico_scatter_regras.png
   └─ grafico_heatmap_correlacao.png
```

## Pipeline de processamento
O script `processar_grupo3.py` executa:
1. Filtragem por cidades do Grupo 3 (Belem, Fortaleza, Recife, Curitiba).
2. Remocao dos atributos de cidade.
3. Remocao de atributos com suporte menor ou igual a 2%.
4. Mapeamento dos itens para 22 categorias por palavras-chave.
5. Agregacao binaria por categoria.
6. Remocao de categorias com suporte menor que 10%.
7. Mineracao de itemsets e regras (Apriori).
8. Consolidacao de regras duplicadas (mantendo maior lift).

## Principais resultados
- Atributos apos remover cidades: 733
- Atributos removidos por suporte <= 2%: 427
- Atributos restantes: 306
- Categorias finais (suporte >= 10%): 20
- Regras unicas no grid: 20.748
- Top regras para analise final: 30
- Melhor lift (Top 30): 1.704

## Como executar
### 1) Criar e ativar ambiente virtual (opcional)
No Windows PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Instalar dependencias
```powershell
pip install pandas mlxtend matplotlib seaborn
```

### 3) Executar processamento
```powershell
python processar_grupo3.py
```

### 4) Gerar visualizacoes
```powershell
python gerar_visuais_grupo3.py
```

## Interpretacao dos graficos
- `grafico_suporte_categorias.png`: categorias mais frequentes nas transacoes.
- `grafico_top10_regras_lift.png`: regras mais fortes por lift.
- `grafico_scatter_regras.png`: relacao entre confianca e lift (com suporte por tamanho/cor).
- `grafico_heatmap_correlacao.png`: correlacao entre categorias com maior suporte.

## Integrantes
- Bruno
- Leonardo
- Maria Eduarda

## Observacoes
- Regras de associacao indicam coocorrencia, nao causalidade.
- Itens nao classificados automaticamente estao documentados em `saida_grupo3/atributos_nao_classificados.txt`.
