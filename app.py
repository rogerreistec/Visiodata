# app.py
# VisioData — Estoques e Produção Hemoterápica (Brasil)
# -----------------------------------------------------
# Requisitos: ver requirements.txt de exemplo:
# streamlit, pandas, numpy, pydeck, requests, python-dateutil

import io
import csv
import json
import time
import base64
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st


# =========================
# CONFIG / ESTILO BÁSICO
# =========================
st.set_page_config(
    page_title="VisioData — Painel de Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide"
)

CUSTOM_CSS = """
<style>
/* Cabeçalho do título */
.title-row{display:flex;align-items:center;gap:.5rem}
.title-pill{background:#E10600;color:#fff;padding:.25rem .6rem;border-radius:999px;font-weight:700}
.muted{color:#6B7280}
footer{visibility:hidden}
.block-container{padding-top:1.2rem}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# =========================
# UTILITÁRIOS
# =========================
@st.cache_data(show_spinner=False, ttl=60*60)
def fetch_url_bytes(url: str) -> bytes:
    """Baixa o conteúdo bruto de uma URL (cacheado)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return r.content


def try_read_csv(bytes_buf: bytes) -> pd.DataFrame:
    """
    Leitura robusta de CSV:
    - tenta ; e , com utf-8/latin1
    - tenta C engine e python engine (sem low_memory) para evitar o erro
    - pula linhas ruins
    """
    attempts = [
        dict(sep=";", encoding="utf-8", engine="c"),
        dict(sep=";", encoding="latin1", engine="c"),
        dict(sep=",", encoding="utf-8", engine="c"),
        dict(sep=",", encoding="latin1", engine="c"),
        dict(sep=";", encoding="utf-8", engine="python"),
        dict(sep=";", encoding="latin1", engine="python"),
        dict(sep=",", encoding="utf-8", engine="python"),
        dict(sep=",", encoding="latin1", engine="python"),
    ]
    last_err = None
    for kw in attempts:
        try:
            # OBS: on_bad_lines='skip' ajuda com linhas truncadas;
            # não usar low_memory com engine='python'
            df = pd.read_csv(
                io.BytesIO(bytes_buf),
                on_bad_lines="skip",
                dtype_backend="numpy_nullable",
                **kw
            )
            if df.shape[1] <= 1:
                # Provável separador errado, continue tentando
                last_err = RuntimeError("Separador incorreto (apenas 1 coluna detectada).")
                continue
            return df
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Falha ao ler CSV. Último erro: {last_err}")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = []
    for c in df.columns:
        c2 = str(c).strip().lower()
        c2 = c2.replace("\n", " ")
        c2 = " ".join(c2.split())
        new_cols.append(c2)
    df.columns = new_cols
    return df


def first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    return None


def numeric_candidates(df: pd.DataFrame) -> list[str]:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # também tentar "valor" textual
    more = [c for c in df.columns if any(t in c for t in ["coleta", "uso", "transfus", "valor", "total"])]
    # manter ordem, sem duplicar
    out = []
    for c in num_cols + more:
        if c not in out and c in df.columns:
            out.append(c)
    return out


UF_COORD = {
    "AC": (-9.0238, -70.8120), "AL": (-9.5713, -36.7820), "AM": (-3.4168, -65.8561),
    "AP": (1.3730, -51.8619), "BA": (-12.5797, -41.7007), "CE": (-5.4984, -39.3206),
    "DF": (-15.7998, -47.8645), "ES": (-19.1834, -40.3089), "GO": (-15.8270, -49.8362),
    "MA": (-5.0429, -45.9653), "MG": (-18.5122, -44.5550), "MS": (-20.7722, -54.7852),
    "MT": (-12.6819, -56.9211), "PA": (-3.9731, -52.2500), "PB": (-7.2399, -36.7819),
    "PE": (-8.8137, -36.9541), "PI": (-7.7183, -42.7289), "PR": (-24.4842, -51.8149),
    "RJ": (-22.9110, -43.2094), "RN": (-5.4026, -36.9541), "RO": (-10.83, -63.34),
    "RR": (2.7376, -62.0751), "RS": (-30.0346, -51.2177), "SC": (-27.5935, -48.5585),
    "SE": (-10.5741, -37.3857), "SP": (-23.5505, -46.6333), "TO": (-9.4656, -48.4682)
}


# ==================================
# SIDEBAR / NAVEGAÇÃO / LOGO
# ==================================
with st.sidebar:
    logo_path = Path("assets/logo.png")
    st.markdown("<div class='title-row'><div class='title-pill'>VisioData</div></div>", unsafe_allow_html=True)
    if logo_path.exists():
        st.image(str(logo_path), caption="VisioData", use_column_width=True)
    else:
        st.markdown("**:drop_of_blood: VisioData**", unsafe_allow_html=True)

    page = st.radio(
        "Navegação",
        ["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"],
        index=0
    )

st.markdown("<div class='title-row'><div class='title-pill'>VisioData</div>"
            "<h1>Painel de Estoques e Produção Hemoterápica</h1></div>", unsafe_allow_html=True)
st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")


# ==================================
# PÁGINA: ANVISA
# ==================================
def page_anvisa():
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")

    DEFAULT_URL = (
        "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
        "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
        "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
    )

    url = st.text_input("URL do CSV (Hemoprod — ANVISA)", value=DEFAULT_URL)

    col_btn, col_alt = st.columns([1, 3], vertical_alignment="bottom")
    clicked = col_btn.button("Carregar agora", type="primary")

    with col_alt.expander("…ou envie o CSV (alternativa)"):
        up = st.file_uploader("Upload (CSV até 200MB)", type=["csv"])

    df = None
    if clicked or up is not None:
        try:
            if up is not None:
                bytes_buf = up.read()
            else:
                with st.spinner("Baixando CSV da ANVISA…"):
                    bytes_buf = fetch_url_bytes(url)
            with st.spinner("Processando CSV…"):
                df = try_read_csv(bytes_buf)
                df = normalize_columns(df)

            st.success(f"Base carregada: **{len(df):,}** linhas × **{df.shape[1]}** colunas.".replace(",", "."))
            with st.expander("Amostra (100 linhas)"):
                st.dataframe(df.head(100), use_container_width=True)

        except Exception as e:
            st.error(f"Falha ao carregar: {e}")

    if df is None:
        st.info("Use **Carregar agora** (ou faça upload) para visualizar KPIs e mapa.")
        return

    # ---------- aba KPIs / Mapa ----------
    tab1, tab2 = st.tabs(["KPIs automáticos", "Mapa por UF"])

    with tab1:
        # Detectar colunas de ano/UF
        ano_col = first_col(df, ["ano de referência", "ano_de_referencia", "ano", "ano referencia"])
        uf_col = first_col(df, ["uf", "unidade federativa", "sigla uf", "estado"])

        # Selects com fallback
        ano_col = st.selectbox(
            "Coluna de ano (se não detectar)",
            options=[ano_col] + [c for c in df.columns if c != ano_col],
            index=0 if ano_col in df.columns else 0
        )
        uf_col = st.selectbox(
            "Coluna UF (se existir)",
            options=[uf_col] + [c for c in df.columns if c != uf_col],
            index=0 if uf_col in df.columns else 0
        )

        # Métrica
        num_cands = numeric_candidates(df)
        default_metric = num_cands[0] if num_cands else None
        metric = st.selectbox(
            "Coluna MÉTRICA (numérica)",
            options=[default_metric] + [c for c in df.columns if c != default_metric],
            index=0 if default_metric in df.columns else 0
        )

        # KPIs
        # Se a métrica não for numérica, criaremos uma contagem simples (1 por linha)
        df_kpi = df.copy()
        if metric and (metric in df_kpi.columns) and (pd.api.types.is_numeric_dtype(df_kpi[metric]) is False):
            # tentar converter
            df_kpi[metric] = pd.to_numeric(df_kpi[metric], errors="coerce")
        if not metric or (metric not in df_kpi.columns) or (df_kpi[metric].dropna().empty):
            df_kpi["__valor__"] = 1
            metric = "__valor__"

        # KPIs de cabeçalho
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", f"{len(df_kpi):,}".replace(",", "."))
        c2.metric("UF distintas", f"{df_kpi[uf_col].nunique(dropna=True) if uf_col in df_kpi.columns else 0}")
        c3.metric("Total (métrica)", f"{float(df_kpi[metric].sum()):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))

        # Tabela resumida por UF/ANO
        group_by = []
        if ano_col in df_kpi.columns:
            group_by.append(ano_col)
        if uf_col in df_kpi.columns:
            group_by.append(uf_col)

        if group_by:
            out = (
                df_kpi.groupby(group_by, dropna=True)[metric]
                .sum(numeric_only=True)
                .reset_index()
                .rename(columns={metric: "valor"})
            )
            st.dataframe(out, use_container_width=True, height=420)
        else:
            st.info("Selecione ao menos uma coluna (UF e/ou ano) para agregação.")

    with tab2:
        # Mapa por UF
        ano_map = first_col(df, [ano_col] if ano_col else [])
        uf_map = first_col(df, [uf_col] if uf_col else [])
        ano_map = st.selectbox("Coluna de ano (se não detectar)", options=[ano_map] + [c for c in df.columns if c != ano_map], index=0)
        uf_map = st.selectbox("Coluna UF", options=[uf_map] + [c for c in df.columns if c != uf_map], index=0)
        metric_map = st.selectbox("Coluna de valor (coletas/uso/etc)", options=[metric] + [c for c in df.columns if c != metric], index=0)

        df_map = df.copy()
        # converter métrica para numérico
        if metric_map not in df_map.columns:
            st.warning("Selecione uma coluna de valor válida.")
            return
        df_map[metric_map] = pd.to_numeric(df_map[metric_map], errors="coerce")

        if uf_map not in df_map.columns:
            st.warning("Selecione a coluna UF.")
            return

        if df_map[metric_map].dropna().empty:
            st.warning("A coluna de valor selecionada não é numérica ou está vazia.")
            return

        grp_cols = [uf_map]
        if ano_map in df_map.columns:
            grp_cols.append(ano_map)

        g = df_map.groupby(grp_cols, dropna=True)[metric_map].sum(numeric_only=True).reset_index()
        g = g.rename(columns={metric_map: "valor"})

        pts = []
        for _, row in g.iterrows():
            uf = str(row[uf_map]).strip().upper()
            if uf in UF_COORD:
                lat, lon = UF_COORD[uf]
                pts.append({"position": [lon, lat], "uf": uf, "valor": float(row["valor"])})

        if not pts:
            st.info("Sem dados suficientes para o mapa (verifique UF e métrica).")
            return

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=pts,
            get_position="position",
            get_radius="valor",
            radius_scale=0.05,
            pickable=True,
            get_fill_color=[225, 0, 0, 140],
        )
        view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.2)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{uf}: {valor}"}))


# ==================================
# PÁGINA: ESTOQUES ESTADUAIS
# ==================================
def page_estaduais():
    st.subheader("Estoques estaduais — referências e links úteis")

    st.caption("Atalhos rápidos (busca pública) para páginas de hemocentros / doação por UF.")
    data = []
    for uf, nome in [
        ("AC", "Acre"), ("AL", "Alagoas"), ("AM", "Amazonas"), ("AP", "Amapá"), ("BA", "Bahia"),
        ("CE", "Ceará"), ("DF", "Distrito Federal"), ("ES", "Espírito Santo"), ("GO", "Goiás"),
        ("MA", "Maranhão"), ("MG", "Minas Gerais"), ("MS", "Mato Grosso do Sul"), ("MT", "Mato Grosso"),
        ("PA", "Pará"), ("PB", "Paraíba"), ("PE", "Pernambuco"), ("PI", "Piauí"), ("PR", "Paraná"),
        ("RJ", "Rio de Janeiro"), ("RN", "Rio Grande do Norte"), ("RO", "Rondônia"), ("RR", "Roraima"),
        ("RS", "Rio Grande do Sul"), ("SC", "Santa Catarina"), ("SE", "Sergipe"), ("SP", "São Paulo"),
        ("TO", "Tocantins"),
    ]:
        link_google = f"https://www.google.com/search?q=hemocentro+{uf}+{nome}+doa%C3%A7%C3%A3o+de+sangue"
        data.append({"UF": uf, "Estado": nome, "Pesquisar": link_google})

    df_links = pd.DataFrame(data)
    # Renderiza com links clicáveis
    def as_link(url, text="Abrir"):
        return f"<a href='{url}' target='_blank'>{text} ↗</a>"

    df_links["Pesquisar"] = df_links["Pesquisar"].apply(lambda u: as_link(u, "Buscar"))
    st.write(df_links.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.info("Conectores oficiais (URLs diretas) podem ser adicionados aqui quando você tiver as fontes por UF.")


# ==================================
# PÁGINA: CADASTRAR DOADOR
# ==================================
def page_doador():
    st.subheader("Cadastrar doador(a) potencial")

    with st.form("cad_doador", clear_on_submit=False):
        c1, c2, c3 = st.columns(3)
        nome = c1.text_input("Nome completo *")
        email = c2.text_input("E-mail")
        fone = c3.text_input("Telefone/WhatsApp")

        c4, c5, c6 = st.columns(3)
        uf = c4.selectbox("UF", list(UF_COORD.keys()))
        cidade = c5.text_input("Cidade")
        abo = c6.selectbox("Tipo sanguíneo (ABO)", ["A", "B", "AB", "O"])

        rh = st.radio("Fator RH", ["+", "-"], horizontal=True, index=0)

        aceito = st.checkbox("Autorizo o contato para campanhas de doação.")
        enviar = st.form_submit_button("Salvar cadastro", type="primary")

    if "doadores" not in st.session_state:
        st.session_state["doadores"] = []

    if enviar:
        if not nome:
            st.warning("Informe o **Nome**.")
        else:
            item = dict(
                data=datetime.now().isoformat(timespec="seconds"),
                nome=nome, email=email, fone=fone, uf=uf, cidade=cidade,
                abo=abo, rh=rh, consentimento=bool(aceito)
            )
            st.session_state["doadores"].append(item)
            st.success("Cadastro recebido! Obrigado(a).")

    if st.session_state["doadores"]:
        df_d = pd.DataFrame(st.session_state["doadores"])
        st.write("Registros recebidos:")
        st.dataframe(df_d, use_container_width=True)

        csv_bytes = df_d.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV de doadores", data=csv_bytes, file_name="doadores.csv", mime="text/csv")


# ==================================
# PÁGINA: SOBRE
# ==================================
def page_sobre():
    st.subheader("Sobre")
    st.markdown(
        """
        **VisioData** — painel experimental para visualização de dados de produção hemoterápica
        e estoques por UF.  
        - **Fonte** principal: ANVISA (Hemoprod)  
        - **Uso**: demonstrações acadêmicas e prototipagem
        """
    )
    st.caption("Dica: para produção, configure um agendamento (GitHub Actions/Streamlit) para atualizar a base.")


# =========================
# ROTEAMENTO
# =========================
if page == "ANVISA (nacional)":
    page_anvisa()
elif page == "Estoques estaduais":
    page_estaduais()
elif page == "Cadastrar doador":
    page_doador()
else:
    page_sobre()
