# app.py
# VisioData — Estoques e Produção Hemoterápica (Brasil)

from __future__ import annotations
import io
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

# =========================
# CONFIGURAÇÃO GERAL/ESTILO
# =========================
st.set_page_config(
    page_title="VisioData — Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

CUSTOM_CSS = """
<style>
/* deixa o topo mais limpo */
footer {visibility: hidden;}
/* título com badge */
.title-row {display:flex; align-items:center; gap:.5rem;}
.badge {background:#E10600; color:#fff; padding:.25rem .6rem; border-radius:999px; font-weight:700;}
/* mensagem suave */
.muted {color:#6B7280;}
/* cards métricas */
.kpi-card {padding: 0.75rem 1rem; border: 1px solid #e5e7eb; border-radius: .75rem; background: #fff;}
.kpi-value {font-size: 1.75rem; font-weight: 700;}
.kpi-label {color:#6B7280; font-size: .85rem;}
/* tabela sem quebrar layout */
.block-container {padding-top: 1.5rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# =========
# CONSTANTES
# =========
# localização de logo (corrige: seu repo usa 'ativos/')
LOGO_CANDIDATOS = [Path("ativos/logo.png"), Path("ativos/logo.svg"), Path("assets/logo.png")]

# coordenadas aproximadas de cada UF para o mapa
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

# URL pública atual do Hemoprod (padrão da ANVISA)
DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

# ==========
# SIDE BAR UI
# ==========
with st.sidebar:
    # corrige o carregamento da logo — usa 'ativos/' se existir
    logo_path = next((p for p in LOGO_CANDIDATOS if p.exists()), None)
    if logo_path:
        st.image(str(logo_path), use_container_width=True)
    else:
        st.write("**VisioData**")

    st.markdown("### Navegação")
    page = st.radio(
        "Escolha a seção",
        ["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

# ===================
# FUNÇÕES AUXILIARES
# ===================
@st.cache_data(show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    # leitura robusta (ISO-8859-1 cobre acentos comuns em planilhas do gov.br)
    df = pd.read_csv(url, sep=",", encoding="utf-8", low_memory=False)
    return df

def normaliza_uf(s: pd.Series) -> pd.Series:
    s = s.astype(str).str.strip().str.upper()
    # tenta reduzir nomes completos para sigla (ex.: "São Paulo" -> "SP")
    mapa_nominal = {
        "SAO PAULO": "SP", "SÃO PAULO": "SP", "RIO DE JANEIRO": "RJ", "MINAS GERAIS": "MG",
        "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "PARANA": "PR", "PARANÁ": "PR",
        "RIO GRANDE DO SUL": "RS", "SANTA CATARINA": "SC", "DISTRITO FEDERAL": "DF",
        "GOIAS": "GO", "GOIÁS": "GO", "BAHIA": "BA", "PERNAMBUCO": "PE", "CEARA": "CE", "CEARÁ": "CE",
        "RIO GRANDE DO NORTE": "RN", "PARAIBA": "PB", "PARAÍBA": "PB", "PIAUI": "PI", "PIAUÍ": "PI",
        "MARANHAO": "MA", "MARANHÃO": "MA", "ALAGOAS": "AL", "SERGIPE": "SE", "PARA": "PA", "PARÁ": "PA",
        "AMAPA": "AP", "AMAPÁ": "AP", "AMAZONAS": "AM", "ACRE": "AC", "RONDONIA": "RO", "RONDÔNIA": "RO",
        "RORAIMA": "RR", "MATO GROSSO": "MT", "MATO GROSSO DO SUL": "MS", "TOCANTINS": "TO",
    }
    s = s.replace(mapa_nominal)
    s = s.str.slice(-2)  # se vier "UF-XX", pega o final
    s = s.where(s.isin(UF_VALIDAS), np.nan)
    return s

def coerce_numeric(col: pd.Series) -> pd.Series:
    # transforma qualquer coluna em numérica (erros -> NaN)
    return pd.to_numeric(col, errors="coerce")

def make_kpis_box(df: pd.DataFrame, uf_col: str, ano_col: Optional[str]) -> None:
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.markdown('<div class="kpi-card"><div class="kpi-value">{:,}</div><div class="kpi-label">Registros</div></div>'.format(len(df)), unsafe_allow_html=True)
    with col2:
        num_ufs = df[uf_col].nunique(dropna=True) if uf_col in df.columns else 0
        st.markdown('<div class="kpi-card"><div class="kpi-value">{}</div><div class="kpi-label">UF distintas</div></div>'.format(num_ufs), unsafe_allow_html=True)
    with col3:
        if ano_col and ano_col in df.columns:
            anos = sorted(df[ano_col].dropna().unique().tolist())
            if len(anos) > 0:
                st.markdown('<div class="kpi-card"><div class="kpi-value">{}</div><div class="kpi-label">Anos detectados</div></div>'.format(", ".join(map(str, anos[:6])) + ("…" if len(anos) > 6 else "")), unsafe_allow_html=True)

def desenha_mapa(df: pd.DataFrame, uf_col: str, valor_col: str) -> None:
    # garante UF e métrica válidas
    if uf_col not in df.columns or valor_col not in df.columns:
        st.info("Selecione UF e uma coluna numérica.")
        return
    d = df[[uf_col, valor_col]].copy()
    d[uf_col] = normaliza_uf(d[uf_col])
    d[valor_col] = coerce_numeric(d[valor_col])
    d = d.dropna(subset=[uf_col, valor_col])
    if d.empty:
        st.info("Sem dados suficientes para o mapa (verifique UF e métrica).")
        return

    agg = d.groupby(uf_col, as_index=False)[valor_col].sum()
    # monta os pontos
    pts = []
    for _, row in agg.iterrows():
        uf = row[uf_col]
        if uf in UF_COORD:
            lat, lon = UF_COORD[uf]
            pts.append({"position": [lon, lat], "uf": uf, "valor": float(row[valor_col])})

    if not pts:
        st.info("Sem dados suficientes para o mapa (UFs inválidas).")
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
    r = pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "{uf}: {valor}"})
    st.pydeck_chart(r, use_container_width=True)

# ============
# CONTEÚDO PÁG
# ============
st.markdown('<div class="title-row">🩸 <span class="badge">VisioData</span> <h2 style="margin:0;">Painel de Estoques e Produção Hemoterápica</h2></div>', unsafe_allow_html=True)
st.caption("Fontes oficiais e dados agregados.")

# memória da sessão
if "df" not in st.session_state:
    st.session_state.df = None

# --------------------------
# PÁGINA: ANVISA (nacional)
# --------------------------
if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")
    url = st.text_input("URL do CSV (Hemoprod — ANVISA)", value=DEFAULT_URL)
    col_btn, _, _ = st.columns([1, 2, 2])
    with col_btn:
        if st.button("Carregar agora", type="primary"):
            with st.spinner("Baixando e preparando dados…"):
                try:
                    df = fetch_csv(url)
                    # normaliza headers
                    df.columns = [c.strip().lower() for c in df.columns]
                    st.session_state.df = df
                    st.success("Base carregada com {:,} linhas × {:,} colunas.".format(*df.shape))
                except Exception as e:
                    st.error(f"Falha ao carregar: {e}")

    # upload alternativo
    with st.expander("…ou envie o CSV (alternativa)"):
        up = st.file_uploader("Drag and drop file here", type=["csv"])
        if up is not None:
            try:
                df = pd.read_csv(up, sep=",", encoding="utf-8", low_memory=False)
                df.columns = [c.strip().lower() for c in df.columns]
                st.session_state.df = df
                st.success("Base carregada com {:,} linhas × {:,} colunas.".format(*df.shape))
            except Exception as e:
                st.error(f"Falha no upload: {e}")

    df = st.session_state.df
    if df is None:
        st.info("Use **Carregar agora** para ler o CSV público (ou faça upload).")
        st.stop()

    # ----- Amostra (colapsada p/ não “bugar” a página)
    with st.expander("Amostra (100 linhas)"):
        st.dataframe(df.head(100), use_container_width=True)

    # ======================
    # KPIs automáticos (fix)
    # ======================
    st.markdown("### KPIs automáticos")
    cols = df.columns.tolist()

    # tenta detectar automaticamente colunas prováveis
    ano_sugestoes = [c for c in cols if "ano" in c and "ref" in c]
    uf_sugestoes = [c for c in cols if c in ("uf", "sigla_uf", "estado", "unidade federativa")]
    metrica_sugestoes = [c for c in cols if "quant" in c or "total" in c or "valor" in c or "id" == c]

    ano_col = st.selectbox("Coluna de ano (se não detectar)", options=["(nenhuma)"] + cols, index=(cols.index(ano_sugestoes[0]) + 1) if ano_sugestoes else 0)
    uf_col = st.selectbox("Coluna UF (se existir)", options=cols, index=cols.index(uf_sugestoes[0]) if uf_sugestoes else 0)
    metrica_col = st.selectbox(
        "Coluna MÉTRICA (numérica)",
        options=cols,
        index=cols.index(metrica_sugestoes[0]) if metrica_sugestoes else 0,
        help="Selecione um campo numérico (ex.: coletas, transfusões, qtd, total, valor…)."
    )

    # KPIs (registros/UFs/anos) – agora sem despejar aquele textão na página
    make_kpis_box(df, uf_col=uf_col, ano_col=None if ano_col == "(nenhuma)" else ano_col)

    # ======================
    # Mapa por UF (corrigido)
    # ======================
    st.markdown("### Mapa por UF")
    desenha_mapa(df, uf_col=uf_col, valor_col=metrica_col)

    # pequena ajuda técnica num expander (evita poluir a tela)
    with st.expander("Ajuda / Dicionário (opcional)"):
        st.write("**Dica:** se o mapa não aparecer, confira se **UF** virou sigla de 2 letras e se a **métrica** é numérica.")
        st.write("Colunas detectadas:", ", ".join(cols))

# --------------------------
# PÁGINA: Estoques estaduais
# --------------------------
elif page == "Estoques estaduais":
    st.subheader("Estoques por tipo sanguíneo — Fontes estaduais (upload)")
    st.caption("Envie planilha com colunas: data, uf, hemocentro, tipo, rh, estoque_atual, estoque_minimo.")

    up = st.file_uploader("Enviar CSV estadual", type=["csv"])
    if up is None:
        st.info("Sem arquivo enviado.")
        st.stop()

    try:
        dfe = pd.read_csv(up, sep=",", encoding="utf-8", low_memory=False)
        dfe.columns = [c.strip().lower() for c in dfe.columns]
    except Exception as e:
        st.error(f"Falha ao ler CSV estadual: {e}")
        st.stop()

    # seleção de UF e filtros simples
    uf_col_e = "uf" if "uf" in dfe.columns else st.selectbox("Escolha a coluna de UF", dfe.columns)
    dfe[uf_col_e] = normaliza_uf(dfe[uf_col_e])

    st.markdown("#### Tabela")
    st.dataframe(dfe, use_container_width=True)

    st.markdown("#### Links oficiais por estado")
    # (mantido) — exiba links clicáveis para páginas estaduais se você já os tinha mapeado.
    links_estado = {
        "SP": "https://prosangue.sp.gov.br/",
        "RJ": "https://www.hemorio.rj.gov.br/",
        "MG": "https://www.hemominas.mg.gov.br/",
        "ES": "https://hemoes.es.gov.br/",
        # acrescente os demais quando tiver as fontes oficiais
    }
    cols = st.columns(6)
    i = 0
    for uf, link in sorted(links_estado.items()):
        with cols[i % 6]:
            st.link_button(f"{uf} • site oficial", link)
        i += 1

# --------------------------
# PÁGINA: Cadastro de doador
# --------------------------
elif page == "Cadastrar doador":
    st.subheader("Cadastro de possíveis doadores")
    st.caption("Coleta local (somente para demonstração).")
    with st.form("form_doador", clear_on_submit=True):
        col1, col2 = st.columns(2)
        nome = col1.text_input("Nome completo *")
        email = col2.text_input("E-mail *")
        col3, col4 = st.columns([1,1])
        uf = col3.selectbox("UF *", sorted(UF_VALIDAS))
        tipo = col4.selectbox("Tipo sanguíneo *", ["A", "B", "AB", "O"])
        rh = st.radio("Fator RH", ["+", "-"], horizontal=True)
        aceite = st.checkbox("Aceito ser contatado(a) para doação", value=True)
        ok = st.form_submit_button("Salvar cadastro", type="primary")

    if ok:
        if not (nome and email):
            st.error("Preencha nome e e-mail.")
        else:
            lista = st.session_state.get("cadastros", [])
            lista.append({"nome": nome, "email": email, "uf": uf, "tipo": tipo+rh, "aceite": aceite})
            st.session_state["cadastros"] = lista
            st.success("Cadastro salvo!")

    if "cadastros" in st.session_state and st.session_state["cadastros"]:
        st.markdown("#### Registros locais (demo)")
        st.dataframe(pd.DataFrame(st.session_state["cadastros"]), use_container_width=True)

# --------------------------
# PÁGINA: Sobre
# --------------------------
else:
    st.subheader("Sobre")
    st.write(
        """
        **VisioData** — painel para análise rápida de produção/estoques hemoterápicos.
        - ANVISA (Hemoprod) com carregamento direto por URL ou upload.
        - KPIs rápidos, mapa por UF e links estaduais.
        - Cadastro local de doadores (demonstração).

        **Créditos de dados:** ANVISA, e sites oficiais dos hemocentros estaduais.
        """
    )
