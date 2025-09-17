# app.py
# VisioData — Estoques e Produção Hemoterápica (Brasil)

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# =========================
# CONFIG / ESTILO
# =========================
st.set_page_config(
    page_title="VisioData — Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

CUSTOM_CSS = """
<style>
footer {visibility: hidden;}
.title-row {display:flex; align-items:center; gap:.5rem; flex-wrap: wrap;}
.badge {background:#E10600; color:#fff; padding:.25rem .6rem; border-radius:999px; font-weight:700;}
.kpi-card {padding:.75rem 1rem; border:1px solid #e5e7eb; border-radius:.75rem; background:#fff;}
.kpi-value {font-size:1.6rem; font-weight:700;}
.kpi-label {color:#6B7280; font-size:.85rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========================
# CONSTANTES / DADOS AUX
# =========================
LOGO_CANDIDATOS = [Path("ativos/logo.png"), Path("ativos/logo.svg"), Path("assets/logo.png")]

UF_COORD = {
    "AC": (-9.02, -70.81), "AL": (-9.57, -36.78), "AP": (0.04, -51.07), "AM": (-3.07, -60.02),
    "BA": (-12.97, -38.50), "CE": (-3.72, -38.54), "DF": (-15.79, -47.88), "ES": (-19.39, -40.07),
    "GO": (-16.68, -49.25), "MA": (-2.53, -44.30), "MT": (-15.60, -56.10), "MS": (-20.51, -54.54),
    "MG": (-19.92, -43.94), "PA": (-1.45, -48.49), "PB": (-7.12, -34.86), "PR": (-25.43, -49.27),
    "PE": (-8.05, -34.90), "PI": (-5.09, -42.80), "RJ": (-22.90, -43.20), "RN": (-5.81, -35.21),
    "RS": (-30.03, -51.23), "RO": (-8.76, -63.90), "RR": (2.82, -60.67), "SC": (-27.59, -48.55),
    "SP": (-23.55, -46.63), "SE": (-10.91, -37.07), "TO": (-10.18, -48.33)
}
UF_VALIDAS = set(UF_COORD.keys())

DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

BUSCA = "https://www.google.com/search?q=doe+sangue+{}"
LINKS_UF: Dict[str, str] = {
    "AC": "https://www.hemoacre.ac.gov.br/",
    "AL": "http://www.hemoal.al.gov.br/",
    "AP": "https://www.portal.ap.gov.br/profissionais/saude/hemocentro-do-amapa",
    "AM": "https://www.hemoam.am.gov.br/",
    "BA": "http://www.hemoba.ba.gov.br/",
    "CE": "https://www.hemoce.ce.gov.br/",
    "DF": "https://www.fhb.df.gov.br/",
    "ES": "https://hemoes.es.gov.br/",
    "GO": "https://www.hemocentro.org.br/",
    "MA": "https://www.hemomar.ma.gov.br/",
    "MT": "https://www.saude.mt.gov.br/hemocentro",
    "MS": "http://www.hemosul.ms.gov.br/",
    "MG": "https://www.hemominas.mg.gov.br/",
    "PA": "https://www.hemopa.pa.gov.br/",
    "PB": "https://paraiba.pb.gov.br/diretas/saude/hemocentro",
    "PR": "https://www.saude.pr.gov.br/HEMEPAR",
    "PE": "http://www.hemope.pe.gov.br/",
    "PI": "https://www.saude.pi.gov.br/hemopi",
    "RJ": "https://www.hemorio.rj.gov.br/",
    "RN": "http://www.hemonorte.rn.gov.br/",
    "RS": "https://saude.rs.gov.br/hemocentro",
    "RO": "http://www.rondonia.ro.gov.br/organograma/secretaria-de-estado-da-saude/fhemeron/",
    "RR": "https://www.rr.gov.br/servico/hemoraima",
    "SC": "https://www.hemosc.org.br/",
    "SP": "https://www.prosangue.sp.gov.br/",
    "SE": "https://www.saude.se.gov.br/hemose/",
    "TO": "https://www.to.gov.br/saude/hemorrede/1077",
}
for uf in UF_VALIDAS:
    LINKS_UF.setdefault(uf, BUSCA.format(uf))

# =========================
# FUNÇÕES AUXILIARES
# =========================
def to_numeric(series: pd.Series) -> pd.Series:
    # troca vírgula decimal por ponto antes de converter
    s = series.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")

@st.cache_data(show_spinner=False)
def fetch_csv_robusto(url: str) -> Tuple[pd.DataFrame, str, str]:
    """
    Tenta várias combinações (sep/encoding) para ler CSV 'problemáticos' da ANVISA.
    Retorna (df, sep_usado, encoding_usado).
    """
    tentativas = [
        (None, "utf-8"), (None, "latin1"),
        (";", "utf-8"), (";", "latin1"),
        (",", "utf-8"), (",", "latin1"),
        ("\t", "utf-8"), ("\t", "latin1"),
    ]
    erros = []
    for sep, enc in tentativas:
        try:
            df = pd.read_csv(
                url,
                sep=sep,
                encoding=enc,
                engine="python",           # mais tolerante
                on_bad_lines="skip",        # ignora linhas ruins
                low_memory=False,
            )
            if df.shape[1] == 1 and df.columns.size == 1:
                # pode ter lido tudo em uma coluna — tenta novamente com outro sep
                raise ValueError("Leitura degenerada (1 coluna).")
            return df, str(sep), enc
        except Exception as e:
            erros.append(f"[sep={sep} enc={enc}] {e}")
    raise RuntimeError("Falha ao ler CSV. Tentativas:\n" + "\n".join(erros))

def normaliza_uf(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    mapa = {
        "SAO PAULO": "SP", "SÃO PAULO": "SP", "RIO DE JANEIRO": "RJ", "MINAS GERAIS": "MG",
        "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "PARANA": "PR", "PARANÁ": "PR",
        "RIO GRANDE DO SUL": "RS", "SANTA CATARINA": "SC", "DISTRITO FEDERAL": "DF",
        "GOIAS": "GO", "GOIÁS": "GO", "BAHIA": "BA", "PERNAMBUCO": "PE", "CEARA": "CE", "CEARÁ": "CE",
        "RIO GRANDE DO NORTE": "RN", "PARAIBA": "PB", "PARAÍBA": "PB", "PIAUI": "PI", "PIAUÍ": "PI",
        "MARANHAO": "MA", "MARANHÃO": "MA", "ALAGOAS": "AL", "SERGIPE": "SE",
        "PARA": "PA", "PARÁ": "PA", "AMAPA": "AP", "AMAPÁ": "AP", "AMAZONAS": "AM", "ACRE": "AC",
        "RONDONIA": "RO", "RONDÔNIA": "RO", "RORAIMA": "RR", "MATO GROSSO DO SUL": "MS",
        "MATO GROSSO": "MT", "TOCANTINS": "TO",
    }
    s = s.replace(mapa)
    s = s.str[-2:]
    s = s.where(s.isin(UF_VALIDAS), np.nan)
    return s

def pick_numeric_column(df: pd.DataFrame, prefer: List[str]) -> Optional[str]:
    cols = df.columns.tolist()
    for p in prefer:
        if p in cols:
            ser = to_numeric(df[p])
            if ser.notna().sum() > 0:
                return p
    for c in cols:
        ser = to_numeric(df[c])
        if ser.notna().sum() > 0:
            return c
    return None

def draw_kpis(df: pd.DataFrame, uf_col: Optional[str], ano_col: Optional[str]):
    c1, c2, c3 = st.columns([1,1,2])
    with c1:
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{len(df):,}</div>'
            '<div class="kpi-label">Registros</div></div>', unsafe_allow_html=True
        )
    with c2:
        ufs = df[uf_col].nunique(dropna=True) if uf_col and uf_col in df.columns else 0
        st.markdown(
            f'<div class="kpi-card"><div class="kpi-value">{ufs}</div>'
            '<div class="kpi-label">UF distintas</div></div>', unsafe_allow_html=True
        )
    with c3:
        if ano_col and ano_col in df.columns:
            anos = pd.to_numeric(df[ano_col], errors="coerce").dropna().astype(int)
            if not anos.empty:
                texto = f"{anos.min()}–{anos.max()}" if anos.nunique() > 1 else f"{anos.iloc[0]}"
            else:
                texto = "—"
            st.markdown(
                f'<div class="kpi-card"><div class="kpi-value">{texto}</div>'
                '<div class="kpi-label">Período</div></div>', unsafe_allow_html=True
            )

def draw_map(df: pd.DataFrame, uf_col: str, metric_col: str):
    if uf_col not in df.columns or metric_col not in df.columns:
        st.info("Selecione corretamente a coluna de UF e a coluna métrica.")
        return
    d = df[[uf_col, metric_col]].copy()
    d[uf_col] = normaliza_uf(d[uf_col])
    d[metric_col] = to_numeric(d[metric_col])
    d = d.dropna(subset=[uf_col, metric_col])
    if d.empty:
        st.info("Sem dados suficientes para o mapa (verifique UF e métrica).")
        return
    agg = d.groupby(uf_col, as_index=False)[metric_col].sum()
    pts = []
    for _, row in agg.iterrows():
        uf = row[uf_col]
        if uf in UF_COORD:
            lat, lon = UF_COORD[uf]
            pts.append({"position": [lon, lat], "uf": uf, "valor": float(row[metric_col])})
    if not pts:
        st.info("Sem pontos válidos para exibir.")
        return
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=pts,
        get_position="position",
        get_radius="valor",
        radius_scale=0.5,
        radius_min_pixels=5,
        pickable=True,
    )
    view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3.2)
    deck = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{uf}: {valor}"})
    st.pydeck_chart(deck, use_container_width=True)

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    logo = next((p for p in LOGO_CANDIDATOS if p.exists()), None)
    if logo: st.image(str(logo), use_container_width=True)
    else: st.write("**VisioData**")

    st.markdown("### Navegação")
    page = st.radio("Escolha a seção", ["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

# =========================
# HEADER
# =========================
st.markdown(
    '<div class="title-row">🩸 <span class="badge">VisioData</span>'
    '<h2 style="margin:0;">Painel de Estoques e Produção Hemoterápica</h2></div>',
    unsafe_allow_html=True
)

if "df_anvisa" not in st.session_state:
    st.session_state.df_anvisa = None

# =========================
# ANVISA (nacional)
# =========================
if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")
    url = st.text_input("URL do CSV (Hemoprod — ANVISA)", value=DEFAULT_URL)
    if st.button("Carregar agora", type="primary"):
        with st.spinner("Carregando…"):
            try:
                df_raw, sep_usado, enc_usado = fetch_csv_robusto(url)
                df = df_raw.copy()
                df.columns = [c.strip().lower() for c in df.columns]
                st.session_state.df_anvisa = df
                st.success(
                    f"Base carregada com {df.shape[0]:,} linhas × {df.shape[1]:,} colunas. "
                    f"(sep={sep_usado}, encoding={enc_usado})"
                )
            except Exception as e:
                st.error(f"Falha ao carregar: {e}")

    with st.expander("…ou envie o CSV (alternativa)"):
        up = st.file_uploader("CSV nacional (alternativo)", type=["csv"], key="up_nac")
        if up is not None:
            try:
                # Também robusto no upload
                df = pd.read_csv(up, sep=None, engine="python", on_bad_lines="skip", low_memory=False)
                df.columns = [c.strip().lower() for c in df.columns]
                st.session_state.df_anvisa = df
                st.success(f"Base carregada com {df.shape[0]:,} linhas × {df.shape[1]:,} colunas.")
            except Exception as e:
                st.error(f"Falha no upload: {e}")

    df = st.session_state.df_anvisa
    if df is None:
        st.info("Use **Carregar agora** (ou faça upload) para visualizar KPIs e mapa.")
        st.stop()

    with st.expander("Amostra (100 linhas)"):
        st.dataframe(df.head(100), use_container_width=True)

    st.markdown("### KPIs automáticos")
    cols = df.columns.tolist()
    ano_sug = next((c for c in cols if "ano" in c and "ref" in c), None)
    uf_sug = "uf" if "uf" in cols else next((c for c in cols if "sigla" in c and "uf" in c), cols[0] if cols else None)

    ano_col = st.selectbox("Coluna de ano (opcional)", ["(nenhuma)"] + cols, index=(cols.index(ano_sug)+1) if ano_sug else 0)
    uf_col = st.selectbox("Coluna UF", cols, index=cols.index(uf_sug) if uf_sug else 0)

    prefer = [c for c in ["quantidade", "qtd", "total", "valor", "transfusoes", "transfusões"] if c in cols]
    metrica_sug = pick_numeric_column(df, prefer) or cols[0]
    metrica_col = st.selectbox("Coluna MÉTRICA (numérica)", cols, index=cols.index(metrica_sug) if metrica_sug in cols else 0)

    draw_kpis(df, uf_col=uf_col, ano_col=None if ano_col == "(nenhuma)" else ano_col)

    st.markdown("### Mapa por UF")
    draw_map(df, uf_col=uf_col, metric_col=metrica_col)

# =========================
# ESTOQUES ESTADUAIS
# =========================
elif page == "Estoques estaduais":
    st.subheader("Estoques por tipo sanguíneo — Fontes estaduais (upload)")

    up = st.file_uploader("Enviar CSV estadual", type=["csv"], key="up_est")
    if up is None:
        st.info("Sem arquivo enviado.")
    else:
        try:
            dfe = pd.read_csv(up, sep=None, engine="python", on_bad_lines="skip", low_memory=False)
            dfe.columns = [c.strip().lower() for c in dfe.columns]
        except Exception as e:
            st.error(f"Falha ao ler CSV estadual: {e}")
            dfe = None

        if dfe is not None:
            with st.expander("Amostra (100 linhas)"):
                st.dataframe(dfe.head(100), use_container_width=True)

            cols = dfe.columns.tolist()
            uf_col_e = "uf" if "uf" in cols else st.selectbox("Coluna UF", cols)
            dfe[uf_col_e] = normaliza_uf(dfe[uf_col_e])

            ano_e = next((c for c in cols if "ano" in c), None)
            ano_col_e = st.selectbox("Coluna de ano (opcional)", ["(nenhuma)"] + cols, index=(cols.index(ano_e)+1) if ano_e else 0)

            prefer_e = [c for c in ["estoque_atual", "estoque_total", "qtd", "quantidade", "total", "valor"] if c in cols]
            metrica_e = pick_numeric_column(dfe, prefer_e) or cols[0]
            metrica_col_e = st.selectbox("Coluna MÉTRICA (numérica)", cols, index=cols.index(metrica_e) if metrica_e in cols else 0)

            st.markdown("### KPIs")
            draw_kpis(dfe, uf_col=uf_col_e, ano_col=None if ano_col_e == "(nenhuma)" else ano_col_e)

            st.markdown("### Mapa por UF")
            draw_map(dfe, uf_col=uf_col_e, metric_col=metrica_col_e)

    st.markdown("### Links oficiais por UF")
    grid = st.columns(6)
    i = 0
    for uf in sorted(UF_VALIDAS):
        with grid[i % 6]:
            st.link_button(f"{uf} • site/contato", LINKS_UF[uf])
        i += 1

# =========================
# CADASTRAR DOADOR
# =========================
elif page == "Cadastrar doador":
    st.subheader("Cadastro de possíveis doadores (demo local)")
    with st.form("form_doador", clear_on_submit=True):
        c1, c2 = st.columns(2)
        nome = c1.text_input("Nome *")
        email = c2.text_input("E-mail *")
        c3, c4 = st.columns(2)
        uf = c3.selectbox("UF *", sorted(UF_VALIDAS))
        tipo = c4.selectbox("Tipo sanguíneo *", ["A", "B", "AB", "O"])
        rh = st.radio("Fator RH", ["+", "-"], horizontal=True)
        aceite = st.checkbox("Aceito ser contatado(a) para doação", value=True)
        ok = st.form_submit_button("Salvar cadastro", type="primary")

    if ok:
        if not (nome and email):
            st.error("Preencha nome e e-mail.")
        else:
            cad = st.session_state.get("cadastros", [])
            cad.append({"nome": nome, "email": email, "uf": uf, "tipo": f"{tipo}{rh}", "aceite": aceite})
            st.session_state["cadastros"] = cad
            st.success("Cadastro salvo!")

    if st.session_state.get("cadastros"):
        st.dataframe(pd.DataFrame(st.session_state["cadastros"]), use_container_width=True)

# =========================
# SOBRE
# =========================
else:
    st.subheader("Sobre")
    st.write(
        """
        **VisioData** — painel leve para explorar dados públicos de produção e estoques hemoterápicos.

        **Recursos**  
        • Carregamento robusto do **Hemoprod (ANVISA)** por URL + upload  
        • **KPIs** e **Mapa por UF** (ANVISA e estadual)  
        • **Links por UF** para hemocentros  
        • Cadastro local (demo) de possíveis doadores
        """
    )
