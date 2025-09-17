# app.py
# VisioData – Painel de Estoques e Produção Hemoterápica
# ------------------------------------------------------
# Autor: você :)
# Observação: este arquivo foi escrito para rodar no Streamlit Cloud e localmente.

from __future__ import annotations
import io
import json
import time
import unicodedata
from typing import Dict, Tuple

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import pydeck as pdk

try:
    import chardet
    HAVE_CHARDET = True
except Exception:
    HAVE_CHARDET = False


# =========================
# --------- UX ------------
# =========================
st.set_page_config(
    page_title="VisioData – Estoques & Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

# ----- Sidebar -----
with st.sidebar:
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("### 🩸")
    with col2:
        st.markdown("### **VisioData**")

    st.caption("Fontes oficiais e dados agregados — prontos para apresentação acadêmica.")


# =========================
# --------- UTILS ---------
# =========================
UF_LATLON: Dict[str, Tuple[float, float]] = {
    "AC": (-9.97499, -67.8243), "AL": (-9.66599, -35.7350), "AP": (0.034934, -51.0694),
    "AM": (-3.11866, -60.0212), "BA": (-12.9718, -38.5011), "CE": (-3.71722, -38.5434),
    "DF": (-15.7939, -47.8828), "ES": (-20.3155, -40.3128), "GO": (-16.6864, -49.2643),
    "MA": (-2.52972, -44.3028), "MT": (-15.6010, -56.0974), "MS": (-20.4697, -54.6201),
    "MG": (-19.9167, -43.9345), "PA": (-1.45583, -48.5039), "PB": (-7.11509, -34.8641),
    "PR": (-25.4284, -49.2733), "PE": (-8.04756, -34.8771), "PI": (-5.08921, -42.8016),
    "RJ": (-22.9068, -43.1729), "RN": (-5.79500, -35.2094), "RS": (-30.0346, -51.2177),
    "RO": (-8.76116, -63.9039), "RR": (2.81972, -60.6733), "SC": (-27.5949, -48.5482),
    "SP": (-23.5505, -46.6333), "SE": (-10.9111, -37.0717), "TO": (-10.1847, -48.3336),
}

# Links por UF (página informativa genérica – pode trocar depois por um endpoint oficial)
LINK_NACIONAL = "https://www.gov.br/anvisa/pt-br/assuntos/sangue-tecidos-celulas-e-orgaos"
LINK_UF = {uf: f"https://www.google.com/search?q=hemocentro+{uf}+estoques" for uf in UF_LATLON}


def slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return (
        s.lower()
        .replace("'", "")
        .replace('"', "")
        .replace("(", "")
        .replace(")", "")
        .replace("/", " ")
        .replace("\\", " ")
    )


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
                         errors="coerce")


# =========================
# ------- CARREGAR --------
# =========================
DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/sangue-tecidos-celulas-e-orgaos/"
    "producao-e-avaliacao-de-servicos-de-hemoterapia/dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

st.markdown("## Painel de Estoques e Produção Hemoterápica — **ANVISA (Hemoprod)**")

url = st.text_input(
    "URL do CSV (Hemoprod — ANVISA)",
    value=DEFAULT_URL,
    key="url_input",
    help="Informe a URL direta para o CSV da ANVISA.",
)

# cache de dados com invalidação por URL e carimbo de tempo (para evitar cache preso no Cloud)
@st.cache_data(show_spinner=False, ttl=60 * 15)
def read_csv_robusto(url_str: str) -> pd.DataFrame:
    # 1) tenta CSV "clássico" utf-8 ; ; ,
    sep_candidates = [";", ",", "\t"]
    enc_candidates = ["utf-8", "latin1", "utf-16", "cp1252"]

    # chardet só quando baixar o conteúdo
    content: bytes | None = None
    try:
        import urllib.request as ur
        with ur.urlopen(url_str, timeout=30) as resp:
            content = resp.read()
    except Exception as e:
        raise RuntimeError(f"Falha ao baixar a URL.\nDetalhes: {e}")

    if content is None or len(content) < 5:
        raise RuntimeError("Conteúdo vazio/pequeno na URL informada.")

    # Se chardet disponível, detecte melhor a codificação
    if HAVE_CHARDET:
        det = chardet.detect(content)
        enc_candidates = [det.get("encoding") or "utf-8"] + enc_candidates

    for enc in enc_candidates:
        for sep in sep_candidates:
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep, low_memory=False)
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue

    # 2) fallback: tenta ler como Excel
    try:
        xls = pd.read_excel(io.BytesIO(content))
        if xls.shape[1] >= 2:
            return xls
    except Exception:
        pass

    raise RuntimeError("Não foi possível ler o arquivo como CSV nem como Excel com os fallbacks.")


c1, c2, c3 = st.columns([2, 1, 2])
with c2:
    carregar = st.button("Carregar agora", type="primary", use_container_width=True, key="btn_carregar")

# Para forçar atualização mesmo com cache, o parâmetro muda (timestamp discreto)
if carregar:
    # Truque simples: a chave de cache muda a cada clique
    st.session_state["cache_buster"] = int(time.time())

if "cache_buster" not in st.session_state:
    st.session_state["cache_buster"] = 0

df = None
erro_carregar = None
if carregar or st.session_state["cache_buster"]:
    with st.spinner("Baixando e processando dados..."):
        try:
            # incluir bust no argumento apenas para cache (não altera a URL real)
            df = read_csv_robusto(url + f"?_={st.session_state['cache_buster']}")
        except Exception as e:
            erro_carregar = str(e)

if erro_carregar:
    st.error(
        "Falha ao carregar o CSV. "
        "Verifique a URL (deve ser o link direto para o arquivo). "
        f"Detalhes: {erro_carregar}"
    )
    st.stop()

if df is None:
    st.info("Use **Carregar agora** para visualizar KPIs e mapa.")
    st.stop()


# =========================
# ---- NORMALIZAÇÃO -------
# =========================
df_original_cols = df.columns.tolist()
df.columns = [slug(c) for c in df.columns]

# tentativas razoáveis de nomes de colunas
poss_ano = [c for c in df.columns if "ano" in c and "refer" in c] or \
           [c for c in df.columns if c in {"ano", "ano_referencia", "ano_de_referencia"}]

poss_uf = [c for c in df.columns if c in {"uf", "unidade_federativa", "estado", "sigla_uf"}] or \
          [c for c in df.columns if c.endswith("_uf") or c.startswith("uf_")]

# heurística de coluna numérica útil (primeira com variação e >=95% numérico)
def sugestao_metrica(_df: pd.DataFrame) -> str | None:
    for c in _df.columns:
        s = to_numeric(_df[c])
        if s.notna().mean() > 0.95 and s.nunique(dropna=True) > 5:
            return c
    return None

sug_metrica = sugestao_metrica(df) or (df.select_dtypes(include=[np.number]).columns.tolist()[:1] or [None])[0]


# =========================
# --------- KPIs ----------
# =========================
st.markdown("### KPIs automáticos")
k1, k2, k3 = st.columns(3)

with k1:
    ano_col = st.selectbox(
        "Coluna de ano (se não detectar)",
        [None] + poss_ano + df.columns.tolist(),
        index=0 if not poss_ano else 1,
        key="sel_ano",
        help="Escolha a coluna que representa o ano de referência.",
    )
with k2:
    uf_col = st.selectbox(
        "Coluna UF (se existir)",
        [None] + poss_uf + df.columns.tolist(),
        index=0 if not poss_uf else 1,
        key="sel_uf",
        help="Escolha a coluna UF/sigla do estado, se existir.",
    )
with k3:
    metr_col = st.selectbox(
        "Coluna MÉTRICA (numérica)",
        df.columns.tolist(),
        index=df.columns.tolist().index(sug_metrica) if (sug_metrica in df.columns) else 0,
        key="sel_metrica",
        help="Selecione a coluna numérica para cálculos/visuais.",
    )

df_view = df.copy()
if ano_col:
    df_view[ano_col] = to_numeric(df_view[ano_col]).astype("Int64")

df_view[metr_col] = to_numeric(df_view[metr_col])

reg_total = int(len(df_view))
anos_distintos = int(df_view[ano_col].nunique()) if ano_col else 0
ufs_distintas = int(df_view[uf_col].nunique()) if uf_col in df_view.columns else 0

cK1, cK2, cK3 = st.columns(3)
cK1.metric("Registros", f"{reg_total:,}".replace(",", "."))
cK2.metric("Anos distintos", anos_distintos if anos_distintos else "—")
cK3.metric("UF distintas", ufs_distintas if ufs_distintas else "—")

# Tabela rápida do top por UF (se existir)
if uf_col:
    grp = (
        df_view.groupby(uf_col, as_index=False)[metr_col]
        .sum(numeric_only=True)
        .sort_values(metr_col, ascending=False)
        .head(15)
    )
    st.dataframe(grp, use_container_width=True, hide_index=True)

# =========================
# --------- MAPA ----------
# =========================
st.markdown("### Mapa por UF")

cA, cB, cC = st.columns(3)
with cA:
    ano_filtro = st.selectbox(
        "Filtrar ano (opcional)",
        [None] + (sorted(df_view[ano_col].dropna().unique().tolist()) if ano_col else []),
        key="map_ano",
        help="Selecione um ano específico ou deixe vazio para usar tudo.",
    )
with cB:
    agg_op = st.selectbox(
        "Agregação",
        ["soma", "média", "mediana", "máximo", "mínimo"],
        index=0,
        key="map_agg",
    )
with cC:
    st.markdown(" ")  # espaçamento
    st.markdown(f"[🔗 Link nacional da ANVISA]({LINK_NACIONAL})")

if ano_filtro is not None and ano_col:
    df_map = df_view.loc[df_view[ano_col] == ano_filtro].copy()
else:
    df_map = df_view.copy()

if not uf_col:
    st.warning("Nenhuma coluna de UF foi selecionada. Selecione a coluna UF para renderizar o mapa.")
else:
    df_map[uf_col] = df_map[uf_col].astype(str).str.upper().str[:2]

    if agg_op == "soma":
        agg = df_map.groupby(uf_col, as_index=False)[metr_col].sum(numeric_only=True)
    elif agg_op == "média":
        agg = df_map.groupby(uf_col, as_index=False)[metr_col].mean(numeric_only=True)
    elif agg_op == "mediana":
        agg = df_map.groupby(uf_col, as_index=False)[metr_col].median(numeric_only=True)
    elif agg_op == "máximo":
        agg = df_map.groupby(uf_col, as_index=False)[metr_col].max(numeric_only=True)
    else:
        agg = df_map.groupby(uf_col, as_index=False)[metr_col].min(numeric_only=True)

    # acrescenta coordenadas
    agg["lat"] = agg[uf_col].map(lambda u: UF_LATLON.get(u, (np.nan, np.nan))[0])
    agg["lon"] = agg[uf_col].map(lambda u: UF_LATLON.get(u, (np.nan, np.nan))[1])
    agg = agg.dropna(subset=["lat", "lon"])

    # normalização de tamanho
    v = agg[metr_col].copy()
    if v.max() == v.min():
        size = np.full_like(v, 10, dtype=float)
    else:
        size = 5 + 35 * (v - v.min()) / (v.max() - v.min())
    agg["size"] = size

    # pydeck scatter
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=agg,
        get_position="[lon, lat]",
        get_radius="size * 20000",
        get_fill_color=[200, 30, 30, 160],
        pickable=True,
        auto_highlight=True,
    )
    view_state = pdk.ViewState(latitude=-14.3, longitude=-51.7, zoom=3.5)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": f"{uf_col}: {{ {uf_col} }}\n{metr_col}: {{ {metr_col} }}"})
    st.pydeck_chart(r, use_container_width=True)

    # tabela de links por UF
    st.subheader("Links rápidos por UF")
    link_df = pd.DataFrame({
        "UF": agg[uf_col],
        "Valor": agg[metr_col],
        "Link": agg[uf_col].map(LINK_UF),
    }).sort_values("UF")
    st.dataframe(link_df, use_container_width=True, hide_index=True)

# =========================
# ----- CADASTRO DOADOR ----
# =========================
st.markdown("---")
st.markdown("### 🧑‍⚕️ Cadastrar doador (opcional)")
with st.form("form_doador", clear_on_submit=False):
    colA, colB, colC = st.columns(3)
    with colA:
        nome = st.text_input("Nome completo", key="fd_nome")
        email = st.text_input("E-mail", key="fd_email")
    with colB:
        tel = st.text_input("Telefone/WhatsApp", key="fd_tel")
        cidade = st.text_input("Cidade", key="fd_cidade")
    with colC:
        uf = st.selectbox("UF", sorted(UF_LATLON.keys()), key="fd_uf")
        sangue = st.selectbox("Tipo sanguíneo", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key="fd_tipo")
    consent = st.checkbox("Autorizo o uso destes dados para contato sobre doação.", key="fd_consent")
    enviar = st.form_submit_button("Salvar cadastro", type="primary")

if "doadores" not in st.session_state:
    st.session_state["doadores"] = []

if enviar:
    if not (nome and email and consent):
        st.warning("Preencha **nome**, **e-mail** e marque o consentimento.")
    else:
        st.session_state["doadores"].append(
            {"nome": nome, "email": email, "telefone": tel, "cidade": cidade, "uf": uf, "tipo_sanguineo": sangue,
             "ts": pd.Timestamp.utcnow().isoformat()}
        )
        st.success("Cadastro salvo localmente (sessão atual).")

if st.session_state["doadores"]:
    st.markdown("#### Doadores (sessão atual)")
    df_d = pd.DataFrame(st.session_state["doadores"])
    st.dataframe(df_d, use_container_width=True, hide_index=True)
    csv = df_d.to_csv(index=False).encode("utf-8")
    st.download_button("Baixar CSV de doadores", csv, "doadores.csv", "text/csv", use_container_width=True)

# =========================
# --------- RODAPÉ --------
# =========================
with st.expander("Sobre este painel"):
    st.write(
        """
        - **VisioData** agrega dados públicos e oferece visualização rápida para análise acadêmica.
        - Os dados apresentados dependem do arquivo informado (estrutura/nomes podem variar).
        - Links por UF são genéricos (busca) e podem ser trocados por páginas oficiais quando disponíveis.
        """
    )
