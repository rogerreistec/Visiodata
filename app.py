# VisioData – Estoques e Produção Hemoterápica (Brasil)
# ------------------------------------------------------
# Requisitos:
#   pip install streamlit pandas polars duckdb pyarrow pydeck
# Rodar:
#   streamlit run app.py

import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="VisioData – Estoques e Produção Hemoterápica", layout="wide")

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
        st.image("assets/logo.svg", width=120)
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
# FUNÇÕES AUXILIARES
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


# ========================
# PÁGINAS
# ========================
if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")
    st.caption("Dados brutos nacionais a partir de 2022 — CSV oficial da Anvisa.")

    DEFAULT_URL = (
        "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
        "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
        "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
    )
    url = st.text_input("URL do CSV (Hemoprod — ANVISA)", value=DEFAULT_URL)
    col_a, col_b = st.columns([1, 1])
    with col_a:
        go = st.button("Carregar dados ANVISA")
    with col_b:
        up = st.file_uploader("...ou envie o CSV baixado do site", type=["csv"])

    if go or up is not None:
        try:
            # leitura
            df = load_csv_robust(up if up is not None else url)
            df = normalize_columns(df)
            if "uf" in df.columns:
                df["uf"] = coerce_uf(df["uf"])

            st.success(f"Base carregada: {len(df):,} linhas × {len(df.columns)} colunas.")

            # abas
            tab_dados, tab_kpis, tab_mapa = st.tabs(["Dados", "KPIs", "Mapa"])

            # --- Dados
            with tab_dados:
                st.dataframe(df.head(200), use_container_width=True)

            # --- KPIs
            with tab_kpis:
                st.markdown("#### KPIs — escolha tempo/UF e métrica")
                time_col = st.selectbox("Coluna temporal (opcional)", [None] + list(df.columns))
                uf_col = st.selectbox(
                    "Coluna UF (opcional)", [None] + list(df.columns),
                    index=(list(df.columns).index("uf") + 1 if "uf" in df.columns else 0)
                )

                nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                if not nums:
                    for c in df.columns:
                        try:
                            df[c] = pd.to_numeric(
                                df[c].astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
                            )
                        except Exception:
                            pass
                    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
                metric = st.selectbox("Coluna métrica (quantidade)", nums if nums else df.columns)

                group = []
                if time_col:
                    group.append(time_col)
                if uf_col:
                    group.append(uf_col)

                if not group:
                    st.info("Selecione pelo menos UF ou uma coluna temporal.")
                else:
                    kpi = df.groupby(group, dropna=False)[metric].sum(numeric_only=True).reset_index()
                    cA, cB, cC = st.columns(3)
                    with cA:
                        st.metric("Registros", f"{len(df):,}")
                    with cB:
                        st.metric("UF distintas", kpi[uf_col].nunique() if uf_col else 0)
                    with cC:
                        st.metric("Total da métrica", f"{kpi[metric].sum():,.0f}")

                    st.dataframe(kpi.head(200), use_container_width=True)
                    st.download_button(
                        "Baixar resumo (CSV)",
                        data=kpi.to_csv(index=False),
                        file_name="resumo_hemoprod.csv",
                    )

            # --- Mapa
            with tab_mapa:
                st.markdown("#### Mapa por UF (bolhas proporcionais)")
                # reutiliza escolhas se existirem; caso contrário, tenta 'uf' e primeira numérica
                use_metric = metric if "metric" in locals() else (nums[0] if nums else None)
                use_uf = "uf" if "uf" in df.columns else (uf_col if "uf_col" in locals() else None)
                if use_metric is None or use_uf is None:
                    st.info("Volte em KPIs e selecione a coluna UF e uma métrica numérica.")
                else:
                    agg = df.groupby(use_uf)[use_metric].sum(numeric_only=True).reset_index()
                    agg.columns = ["uf", "valor"]
                    agg["uf"] = coerce_uf(agg["uf"])
                    points = []
                    for _, r in agg.iterrows():
                        uf = str(r["uf"]).upper()[:2]
                        if uf in UF_COORD:
                            lat, lon = UF_COORD[uf]
                            points.append({"position": [lon, lat], "uf": uf, "valor": float(r["valor"])})
                    if points:
                        layer = pdk.Layer(
                            "ScatterplotLayer",
                            data=points,
                            get_position="position",
                            get_radius="valor",
                            radius_scale=0.05,
                            radius_min_pixels=5,
                            get_fill_color=[225, 0, 0, 140],
                            pickable=True,
                        )
                        view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.25)
                        st.pydeck_chart(
                            pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{uf}: {valor}"})
                        )
                    else:
                        st.info("Não há UF válidas (use siglas como 'SP', 'RJ').")

        except Exception as e:
            st.error(f"Falha ao carregar: {e}")

elif page == "Estoques estaduais":
    st.subheader("Estoques por tipo sanguíneo — Fontes estaduais (upload)")
    st.caption("Ex.: Fundação Pró-Sangue (SP), HEMORIO (RJ). Integração piloto via upload CSV/Parquet.")

    st.link_button(
        "Abrir Pró-Sangue (SP)",
        "https://www.prosangue.sp.gov.br/noticias/Pr%C3%B3-Sangue%2Bprecisa%2Burgentemente%2Bde%2Bsangue",
    )
    st.link_button("Abrir HEMORIO (RJ)", "https://www.hemorio.rj.gov.br/")

    up = st.file_uploader(
        "Envie um CSV/Parquet com colunas: data, uf, hemocentro, tipo, rh, estoque_atual, estoque_minimo",
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
                alerts = df_e[df_e["cobertura"] < 1.0].copy()

                st.markdown("### ⚠️ Alertas (abaixo do mínimo)")
                if alerts.empty:
                    st.success("Sem alertas no arquivo enviado.")
                else:
                    st.dataframe(alerts.sort_values("cobertura"), use_container_width=True)
                    st.download_button(
                        "Baixar alertas (CSV)",
                        data=alerts.to_csv(index=False),
                        file_name="alertas_estoque.csv",
                    )

                st.markdown("### Mapa de cobertura média por UF")
                cov = df_e.groupby("uf", as_index=False)["cobertura"].mean(numeric_only=True)
                pts = []
                for _, r in cov.iterrows():
                    uf = str(r["uf"]).upper()[:2]
                    if uf in UF_COORD:
                        lat, lon = UF_COORD[uf]
                        pts.append({"position": [lon, lat], "uf": uf, "valor": float(r["cobertura"])})
                if pts:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=pts,
                        get_position="position",
                        get_radius="valor",
                        radius_scale=40,
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
        st.info("Envie um arquivo real de algum estado (SP/RJ, etc.).")

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

**LGPD:** o painel não usa dados pessoais de doadores; apenas indicadores agregados publicados por fontes oficiais.
        """
    )
