# VisioData — Estoques e Produção Hemoterápica (Brasil)
# ------------------------------------------------------
# Rodar local (opcional):  streamlit run app.py
# Depende de: streamlit, pandas, numpy, pydeck, pyarrow, duckdb, polars

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk
from pathlib import Path

st.set_page_config(page_title="VisioData — Estoques e Produção Hemoterápica", layout="wide")

# ========================
# ESTILO / CABEÇALHO
# ========================
CUSTOM_CSS = """
<style>
.title-row { display:flex; align-items:center; gap:.75rem; }
.pill { background:#E10600; color:white; padding:.25rem .6rem; border-radius:999px; font-weight:700; }
.muted { color:#6B7280; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

c1, c2 = st.columns([1, 6])
with c1:
    try:
        st.image("ativos/logo.svg", width=120)
    except Exception:
        st.write(":drop_of_blood:")
with c2:
    st.markdown(
        '<div class="title-row"><span class="pill">VisioData</span>'
        '<h2 style="margin:0">Painel de Estoques e Produção Hemoterápica</h2></div>',
        unsafe_allow_html=True,
    )
    st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

# ========================
# NAVEGAÇÃO
# ========================
st.sidebar.header("Navegação")
page = st.sidebar.radio(
    "Escolha a seção",
    ["ANVISA (nacional)", "Estoques estaduais", "Sobre"],
    index=0,
)

# ========================
# FUNÇÕES / DADOS AUXILIARES
# ========================
@st.cache_data(show_spinner=True)
def load_csv_robust(url_or_file, encodings=("utf-8", "latin-1")):
    """Lê CSV tentando autodetectar separador e encoding."""
    erros = []
    for sep in (None, ";", ","):
        for enc in encodings:
            try:
                df = pd.read_csv(
                    url_or_file,
                    sep=sep,
                    engine=("python" if sep is None else "c"),
                    encoding=enc,
                )
                return df
            except Exception as e:
                erros.append(f"{enc}[sep={sep}]→{e}")
    raise ValueError("Falha ao ler CSV: " + " | ".join(erros))

@st.cache_data(show_spinner=False)
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

# Coordenadas médias das UFs (para pydeck)
UF_COORD = {
    "AC": (-9.02, -70.81), "AL": (-9.62, -36.82), "AM": (-3.47, -60.02), "AP": (0.04, -51.05),
    "BA": (-12.97, -38.51), "CE": (-3.71, -38.54), "DF": (-15.78, -47.93), "ES": (-20.32, -40.34),
    "GO": (-16.68, -49.25), "MA": (-2.53, -44.30), "MG": (-19.92, -43.94), "MS": (-20.48, -54.62),
    "MT": (-15.60, -56.10), "PA": (-1.45, -48.49), "PB": (-7.12, -34.86), "PE": (-8.05, -34.90),
    "PI": (-5.09, -42.80), "PR": (-25.43, -49.27), "RJ": (-22.91, -43.17), "RN": (-5.79, -35.21),
    "RO": (-10.83, -63.34), "RR": (2.82, -60.67), "RS": (-30.03, -51.23), "SC": (-27.59, -48.55),
    "SE": (-10.92, -37.07), "SP": (-23.55, -46.64), "TO": (-10.18, -48.33),
}

NOME2UF = {
    "acre": "AC", "alagoas": "AL", "amazonas": "AM", "amapá": "AP", "bahia": "BA", "ceará": "CE",
    "distrito federal": "DF", "espírito santo": "ES", "goiás": "GO", "maranhão": "MA",
    "minas gerais": "MG", "mato grosso do sul": "MS", "mato grosso": "MT", "pará": "PA",
    "paraíba": "PB", "pernambuco": "PE", "piauí": "PI", "paraná": "PR", "rio de janeiro": "RJ",
    "rio grande do norte": "RN", "rondônia": "RO", "roraima": "RR", "rio grande do sul": "RS",
    "santa catarina": "SC", "sergipe": "SE", "são paulo": "SP", "tocantins": "TO",
}
def coerce_uf(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s_up = s.str.upper()
    ok = s_up.where(s_up.str.len() == 2)
    if ok.isna().any():
        mapped = s.str.lower().map(NOME2UF)
        ok = ok.fillna(mapped)
    return ok

# População estimada (aprox.) para taxas por 100 mil hab.
POP_UF = {
    "AC": 0.906e6, "AL": 3.377e6, "AM": 4.269e6, "AP": 0.877e6, "BA": 14.87e6,
    "CE": 9.24e6, "DF": 3.10e6, "ES": 4.10e6, "GO": 7.29e6, "MA": 6.77e6,
    "MG": 20.5e6, "MS": 2.75e6, "MT": 3.78e6, "PA": 8.69e6, "PB": 4.03e6,
    "PE": 9.68e6, "PI": 3.27e6, "PR": 11.44e6, "RJ": 16.05e6, "RN": 3.50e6,
    "RO": 1.58e6, "RR": 0.73e6, "RS": 10.88e6, "SC": 7.61e6, "SE": 2.36e6,
    "SP": 44.42e6, "TO": 1.61e6
}

def first_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# ========================
# PÁGINAS
# ========================
if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")

    DEFAULT_URL = (
        "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
        "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
        "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
    )

    # se o CSV local existir (atualizado por ação agendada), priorize
    LOCAL_CSV = Path("data/hemoprod_nacional.csv")
    url_inicial = str(LOCAL_CSV) if LOCAL_CSV.exists() else DEFAULT_URL

    url = st.text_input(
        "URL do CSV (Hemoprod — ANVISA)",
        value=url_inicial,
        help="Se a página oficial mudar, cole aqui a nova URL ou use upload.",
    )
    c_a, c_b = st.columns([1, 1])
    with c_a:
        go = st.button("Carregar agora")
    with c_b:
        up = st.file_uploader("...ou envie o CSV (alternativa)", type=["csv"])

    # Auto-load na primeira visita
    if "hemoprod_autoload" not in st.session_state:
        st.session_state["hemoprod_autoload"] = True
        go = True

    if go or up is not None:
        try:
            df = load_csv_robust(up if up is not None else url)
            df = normalize_columns(df)
            if "uf" in df.columns:
                df["uf"] = coerce_uf(df["uf"])

            st.success(f"Base carregada: {len(df):,} linhas × {len(df.columns)} colunas.")
            with st.expander("Amostra (100 linhas)"):
                st.dataframe(df.head(100), use_container_width=True)

            tab_kpis, tab_mapa = st.tabs(["KPIs automáticos", "Mapa por UF"])

            # ================= KPIs automáticos =================
            with tab_kpis:
                ano_col = first_col(df, ["ano", "ano de referência", "ano_referencia"])

                # tenta nomes comuns de métricas; se não, oferece qualquer coluna numérica
                metric_coleta = first_col(df, ["coletas", "qtd_coletas", "coleta_total", "quantidade", "valor"])
                metric_uso    = first_col(df, ["transfusoes", "transfusões", "qtd_transfusoes", "uso_total"])
                num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

                sel_ano = st.selectbox("Coluna de ano (se não detectar)", [ano_col] + [None] + list(df.columns), index=0 if ano_col else 1)
                sel_uf  = st.selectbox("Coluna UF (se existir)", ["uf"] + [None] + list(df.columns), index=0 if "uf" in df.columns else 1)

                default_metric = metric_coleta or metric_uso or (num_cols[0] if num_cols else None)
                options_metric = ([default_metric] + num_cols) if default_metric else num_cols
                sel_metric = st.selectbox("Coluna MÉTRICA (numérica)", options_metric)

                group = []
                if sel_ano: group.append(sel_ano)
                if sel_uf and sel_uf in df.columns: group.append(sel_uf)

                if sel_metric:
                    out = df.groupby(group, dropna=False)[sel_metric].sum(numeric_only=True).reset_index().rename(columns={sel_metric:"valor"})
                    # cartões
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Registros", f"{len(df):,}")
                    with c2: st.metric("UF distintas", out[sel_uf].nunique() if sel_uf in out.columns else 0)
                    with c3: st.metric("Total (métrica)", f"{out['valor'].sum():,.0f}")

                    # Taxa por 100k (se houver UF)
                    if sel_uf and sel_uf in out.columns:
                        out["uf_norm"] = coerce_uf(out[sel_uf])
                        out["taxa_100k"] = out["valor"] / out["uf_norm"].map(POP_UF) * 1e5
                        with st.expander("Taxa por 100 mil hab. (se UF disponível)"):
                            st.dataframe(out[[*group,"valor","taxa_100k"]].head(200), use_container_width=True)
                    else:
                        st.dataframe(out.head(200), use_container_width=True)

                    st.download_button("Baixar resumo (CSV)", out.to_csv(index=False), "resumo_hemoprod.csv")
                else:
                    st.info("Escolha uma coluna numérica para gerar KPIs e mapa.")

            # ================= Mapa por UF =================
            with tab_mapa:
                # usa a mesma métrica escolhida acima; se não existir no escopo, tenta outra numérica
                # (preciso de UF e métrica numérica)
                try:
                    map_col = sel_metric
                except NameError:
                    map_col = num_cols[0] if num_cols else None

                if map_col and "uf" in df.columns:
                    agg = df.groupby("uf")[map_col].sum(numeric_only=True).reset_index()
                    agg["uf"] = coerce_uf(agg["uf"])
                    pts = []
                    for _, r in agg.iterrows():
                        uf = str(r["uf"])[:2].upper()
                        if uf in UF_COORD and pd.notna(r[map_col]):
                            lat, lon = UF_COORD[uf]
                            pts.append({"position":[lon,lat], "uf":uf, "valor": float(r[map_col])})
                    if pts:
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=pts,
                            get_position="position",
                            get_radius="valor",
                            radius_scale=0.05,
                            radius_min_pixels=5,
                            get_fill_color=[225,0,0,140],
                            pickable=True,
                        )
                        view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.2)
                        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{uf}: {valor}"}))
                    else:
                        st.info("Sem dados suficientes para o mapa.")
                else:
                    st.info("Preciso da coluna UF e de uma métrica numérica para o mapa.")

        except Exception as e:
            st.error(f"Falha ao carregar: {e}")
            st.info("Se a URL abrir HTML, baixe o CSV no site e use o upload.")

elif page == "Estoques estaduais":
    st.subheader("Estoques por tipo sanguíneo — Fontes estaduais (upload)")
    st.caption("Use CSV/Parquet com colunas: data, uf, hemocentro, tipo, rh, estoque_atual, estoque_minimo.")

    st.link_button("Pró-Sangue (SP)", "https://www.prosangue.sp.gov.br/noticias/Pr%C3%B3-Sangue%2Bprecisa%2Burgentemente%2Bde%2Bsangue")
    st.link_button("HEMORIO (RJ)", "https://www.hemorio.rj.gov.br/")

    up = st.file_uploader(
        "Envie arquivo real (CSV/Parquet) — SP, RJ, etc.",
        type=["csv", "parquet"],
    )
    if up is not None:
        try:
            if up.name.endswith(".parquet"):
                df_e = pd.read_parquet(up)
            else:
                df_e = load_csv_robust(up)
            df_e = normalize_columns(df_e)
            if "uf" in df_e.columns:
                df_e["uf"] = coerce_uf(df_e["uf"])

            st.dataframe(df_e.head(100), use_container_width=True)

            needed = {"estoque_atual", "estoque_minimo"}
            if needed.issubset(set(df_e.columns)):
                df_e["cobertura"] = df_e["estoque_atual"].astype(float) / df_e["estoque_minimo"].replace(0, np.nan)

                mean_cov = float(df_e["cobertura"].mean(skipna=True))
                abaixo = int((df_e["cobertura"] < 0.8).sum())
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Cobertura média", f"{mean_cov:.2f}x")
                with c2: st.metric("Registros", f"{len(df_e):,}")
                with c3: st.metric("Críticos (<0,8)", abaixo)

                if mean_cov < 0.9 or abaixo > 0:
                    st.warning("🚨 Níveis baixos detectados. Considere agendar uma doação.")
                    st.link_button("Onde doar (Gov.br)", "https://www.gov.br/saude/pt-br/assuntos/saude-de-a-a-z/d/doacao-de-sangue")

                st.markdown("### ⚠️ Lista de alertas (abaixo do mínimo)")
                alerts = df_e[df_e["cobertura"] < 1.0].copy()
                if alerts.empty:
                    st.success("Sem alertas no arquivo enviado.")
                else:
                    st.dataframe(alerts.sort_values("cobertura"), use_container_width=True)
                    st.download_button(
                        "Baixar alertas (CSV)",
                        data=alerts.to_csv(index=False),
                        file_name="alertas_estoque.csv",
                    )

                # Mapa de cobertura média por UF
                if "uf" in df_e.columns:
                    cov = df_e.groupby("uf", as_index=False)["cobertura"].mean(numeric_only=True)
                    pts = []
                    for _, r in cov.iterrows():
                        uf = str(r["uf"]).upper()[:2]
                        if uf in UF_COORD and pd.notna(r["cobertura"]):
                            lat, lon = UF_COORD[uf]
                            pts.append({"position": [lon, lat], "uf": uf, "valor": float(r["cobertura"])})
                    if pts:
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=pts,
                            get_position="position",
                            get_radius="valor",
                            radius_scale=40,  # cobertura ~0..2 → raio fixo/legível
                            radius_min_pixels=5,
                            get_fill_color="[255*(1-valor), 255*valor, 0]",  # vermelho→verde
                            pickable=True,
                        )
                        view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.25)
                        st.pydeck_chart(
                            pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{uf}: cobertura {valor}"})
                        )
            else:
                st.warning("Colunas 'estoque_atual' e 'estoque_minimo' não encontradas.")
        except Exception as e:
            st.error(f"Erro ao processar arquivo estadual: {e}")
    else:
        st.info("Envie um arquivo real (SP/RJ, etc.).")

elif page == "Sobre":
    st.subheader("Sobre o projeto")
    st.markdown(
        """
**VisioData** é um painel em Python + Streamlit para visualização de dados hemoterápicos **agregados** no Brasil.

- **Fontes oficiais**:
  - ANVISA / Hemoprod (CSV nacional a partir de 2022)
  - Hemocentros estaduais (ex.: Fundação Pró-Sangue/SP, HEMORIO/RJ)
- **Tecnologias**: Streamlit, Pandas, PyDeck, DuckDB
- **Objetivo**: facilitar o acesso a informações públicas e incentivar a doação de sangue.

**LGPD:** sem dados pessoais de doadores; apenas indicadores agregados publicados por fontes oficiais.
        """
    )
