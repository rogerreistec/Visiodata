# app.py
# VisioData – Painel de Estoques e Produção Hemoterápica
# Seções: ANVISA (nacional), Estoques estaduais (links), Cadastrar doador, Sobre.
# >>> Correção principal: leitura de CSV sem passar low_memory ao engine="python".

from __future__ import annotations

import io
import os
import textwrap
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# -----------------------------------------------------------------------------
# Configuração básica da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisioData | Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

# Estilinho rápido para o selo VisioData na barra lateral
st.sidebar.markdown(
    """
    <div style="display:flex;gap:.6rem;align-items:center;">
      <span style="font-size:1.3rem;">🩸</span>
      <span style="background:#f43f5e;color:white;font-weight:700;
                   border-radius:10px;padding:.2rem .6rem;display:inline-block;">
        VisioData
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def read_csv_robusto(origem: str | bytes, uploaded: bool = False) -> pd.DataFrame:
    """
    Lê CSV de forma resiliente (URL ou upload). Remove a combinação problemática
    low_memory + engine='python'. Faz inferência de separador e codificação.

    Parameters
    ----------
    origem : str|bytes
        URL (str) ou conteúdo do arquivo (bytes) quando uploaded=True.
    uploaded : bool
        True quando o conteúdo vem de st.file_uploader (bytes/BytesIO).

    Returns
    -------
    pd.DataFrame
    """
    # 1) Origem: caminho ou bytes
    if uploaded:
        byts = origem if isinstance(origem, (bytes, bytearray)) else origem.read()
        buf = io.BytesIO(byts)
    else:
        buf = origem  # URL string

    # 2) Tentativa 1: sep=None (Sniffer) com engine='python'
    try:
        df = pd.read_csv(
            buf,
            sep=None,             # infere o separador
            engine="python",      # mais tolerante a CSVs bagunçados
            on_bad_lines="skip",  # ignora linhas ruins (pandas>=1.4)
            dtype=str             # lê inicialmente tudo como string
        )
        if df.empty:
            raise ValueError("CSV vazio.")
        return df
    except Exception as e1:
        # 3) Tentativa 2: assume ';'
        try:
            if uploaded:
                buf.seek(0)
            df = pd.read_csv(
                buf,
                sep=";",
                engine="python",
                on_bad_lines="skip",
                dtype=str
            )
            if df.empty:
                raise ValueError("CSV vazio.")
            return df
        except Exception as e2:
            # 4) Tentativa 3: assume ','
            try:
                if uploaded:
                    buf.seek(0)
                df = pd.read_csv(
                    buf,
                    sep=",",
                    engine="python",
                    on_bad_lines="skip",
                    dtype=str
                )
                if df.empty:
                    raise ValueError("CSV vazio.")
                return df
            except Exception as e3:
                raise RuntimeError(
                    f"Falha ao ler o CSV. "
                    f"Tentativas: sep=None | ';' | ','\n"
                    f"Erros: {e1}\n{e2}\n{e3}"
                )

def normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas (minúsculas, sem espaços extras)."""
    ren = {c: " ".join(c.strip().lower().split()) for c in df.columns}
    df = df.rename(columns=ren)
    # remove colunas 'unnamed'
    drop_cols = [c for c in df.columns if c.startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df

def detecta_colunas(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Tenta detectar as colunas de ano, uf e lista de métricas numéricas possíveis.
    """
    col_ano = None
    for c in df.columns:
        if "ano" in c and ("ref" in c or "refer" in c or "referência" in c):
            col_ano = c
            break

    col_uf = None
    for c in df.columns:
        if c == "uf" or c.endswith(" uf") or c.startswith("uf "):
            col_uf = c
            break

    # Numericáveis
    possiveis_metricas: List[str] = []
    for c in df.columns:
        # ignora campos muito textuais
        if c in {col_ano, col_uf}:
            continue
        # tenta converter para número rapidamente em amostra
        amostra = pd.to_numeric(df[c].dropna().astype(str).str.replace(",", ".", regex=False),
                                errors="coerce")
        if amostra.notna().sum() > 0:
            possiveis_metricas.append(c)

    # preferência por "id da resposta"
    if "id da resposta" in df.columns and "id da resposta" in possiveis_metricas:
        possiveis_metricas = ["id da resposta"] + [m for m in possiveis_metricas if m != "id da resposta"]

    return col_ano, col_uf, possiveis_metricas

def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce"
    )

# centros aproximados de UF (para o mapa com bolhas)
UF_CENTER = {
    "AC": (-9.0238, -70.8120), "AL": (-9.5713, -36.7820), "AM": (-3.4168, -65.8561),
    "AP": (1.4156, -51.6022),  "BA": (-12.9694, -41.5556), "CE": (-5.4984, -39.3206),
    "DF": (-15.7998, -47.8645), "ES": (-19.6113, -40.1853), "GO": (-15.8270, -49.8362),
    "MA": (-4.9609, -45.2744), "MG": (-18.5122, -44.5550), "MS": (-20.7722, -54.7852),
    "MT": (-12.6819, -55.6370), "PA": (-3.8431, -52.2500),  "PB": (-7.1219, -36.7240),
    "PE": (-8.8137, -36.9541), "PI": (-7.7183, -42.7289),  "PR": (-24.4842, -51.8625),
    "RJ": (-22.1700, -42.0000), "RN": (-5.4026, -36.9541), "RO": (-10.83, -63.34),
    "RR": (2.7376, -62.0751),  "RS": (-29.3344, -53.5000), "SC": (-27.2423, -50.2189),
    "SE": (-10.5741, -37.3857), "SP": (-22.19, -48.79),     "TO": (-10.1753, -48.2982)
}

def kpi_box(label: str, value: int | float | str):
    st.metric(label, value)


# -----------------------------------------------------------------------------
# Seção: ANVISA (nacional)
# -----------------------------------------------------------------------------
def pagina_anvisa():
    st.header("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    with st.expander("URL do CSV (Hemoprod — ANVISA)", expanded=True):
        url_default = (
            "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
            "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
            "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
        )
        url = st.text_input(
            "Cole a URL do CSV aqui",
            url_default,
            placeholder="https://.../hemoprod_nacional.csv",
            key="hemoprod_url",
        )

        colb1, colb2 = st.columns([1, 1])
        with colb1:
            botao = st.button("Carregar agora", type="primary")

        with colb2:
            st.caption("…ou envie o CSV (alternativa)")
            upload = st.file_uploader(" ", type=["csv"], label_visibility="collapsed")

    df: Optional[pd.DataFrame] = None
    erro: Optional[str] = None

    if botao and not url and not upload:
        st.warning("Informe a URL do CSV ou envie um arquivo.")
        return

    if botao or upload is not None:
        with st.spinner("Lendo a base…"):
            try:
                if upload is not None:
                    df = read_csv_robusto(upload.getvalue(), uploaded=True)
                else:
                    df = read_csv_robusto(url, uploaded=False)
            except Exception as e:
                erro = str(e)

    if erro:
        st.error(
            "Falha ao carregar: "
            "verifique a URL/arquivo. Mensagem técnica:\n\n" + erro
        )
        return

    if df is None:
        st.info("Preencha a URL ou envie um CSV e clique **Carregar agora**.")
        return

    # Normaliza nomes e tenta detectar colunas-alvo
    df = normaliza_colunas(df)
    col_ano, col_uf, metricas = detecta_colunas(df)

    st.subheader("KPIs automáticos")

    # Seletor de colunas (com padrões detectados)
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        ano_col = st.selectbox(
            "Coluna de ano (se não detectar)",
            options=["<não há>"] + list(df.columns),
            index=(df.columns.tolist().index(col_ano) + 1) if col_ano in df.columns else 0,
            help="Se o arquivo não tiver coluna de ano, deixe como '<não há>'.",
            key="anvisa_col_ano",
        )
        if ano_col == "<não há>":
            ano_col = None

    with c2:
        uf_col = st.selectbox(
            "Coluna UF (se existe)",
            options=["<não há>"] + list(df.columns),
            index=(df.columns.tolist().index(col_uf) + 1) if col_uf in df.columns else 0,
            help="Selecione a coluna que representa a UF (quando existir).",
            key="anvisa_col_uf",
        )
        if uf_col == "<não há>":
            uf_col = None

    with c3:
        if len(metricas) == 0:
            metricas = [c for c in df.columns if c not in {ano_col, uf_col}]
        met_col = st.selectbox(
            "Coluna MÉTRICA (numérica)",
            options=metricas,
            index=0 if len(metricas) else 0,
            help="Escolha a coluna que será somada/contada nas agregações.",
            key="anvisa_col_met",
        )

    # KPIs
    try:
        total_reg = len(df)
        k1, k2, k3 = st.columns(3)
        with k1:
            kpi_box("Registros", f"{total_reg:,}".replace(",", "."))
        if ano_col:
            anos_distintos = df[ano_col].nunique(dropna=True)
        else:
            anos_distintos = 0
        with k2:
            kpi_box("Anos distintos", anos_distintos)

        if uf_col:
            ufs_distintas = df[uf_col].nunique(dropna=True)
        else:
            ufs_distintas = 0
        with k3:
            kpi_box("UF distintas", ufs_distintas)
    except Exception:
        pass

    # Agregação por UF (e ano mais recente, quando existir)
    df_ag = df.copy()

    # força métrica para número
    df_ag[met_col] = to_numeric_safe(df_ag[met_col])

    if ano_col and df_ag[ano_col].notna().any():
        # usa o ano mais recente como padrão
        try:
            anos_validos = pd.to_numeric(df_ag[ano_col], errors="coerce")
            ano_recente = int(anos_validos.dropna().max())
        except Exception:
            ano_recente = None
        if ano_recente is not None:
            st.caption(f"Agrupando pelo ano mais recente detectado: **{ano_recente}**")
            df_ag = df_ag[df_ag[ano_col].astype(str) == str(ano_recente)]

    if uf_col:
        grupo = df_ag.groupby(uf_col, dropna=False, as_index=False)[met_col].sum()
        grupo = grupo.rename(columns={uf_col: "uf", met_col: "valor"})
        # limpa UFs inválidas
        grupo["uf"] = grupo["uf"].astype(str).str.upper().str.strip()
        grupo = grupo[grupo["uf"].isin(UF_CENTER.keys())]
    else:
        grupo = pd.DataFrame(columns=["uf", "valor"])

    st.subheader("Mapa por UF")
    if len(grupo) == 0:
        st.info("Não há dados suficientes para o mapa (verifique UF e métrica).")
    else:
        # monta a camada de bolhas
        plot_df = []
        for _, r in grupo.iterrows():
            uf = r["uf"]
            val = float(r["valor"]) if pd.notna(r["valor"]) else 0.0
            if uf in UF_CENTER:
                lat, lon = UF_CENTER[uf]
                plot_df.append(dict(uf=uf, valor=val, lat=lat, lon=lon))

        if len(plot_df) == 0:
            st.info("Não foi possível posicionar as UFs no mapa.")
        else:
            plot_df = pd.DataFrame(plot_df)
            # escala para tamanho da bolha
            rad = 5000 + 3000 * np.sqrt(plot_df["valor"].replace(0, np.nan).fillna(0) / (plot_df["valor"].max() or 1))

            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot_df.assign(radius=rad),
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[244, 63, 94, 160],  # vermelho
                pickable=True,
            )

            view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5)
            tooltip = {"text": "{uf}: {valor}"}
            deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip, map_style="dark_no_labels")
            st.pydeck_chart(deck, use_container_width=True)

    st.subheader("Tabela agregada por UF")
    st.dataframe(grupo.sort_values("valor", ascending=False), use_container_width=True)


# -----------------------------------------------------------------------------
# Seção: Estoques estaduais (links oficiais/Google)
# -----------------------------------------------------------------------------
def pagina_links_estaduais():
    st.header("Acesse páginas/oficiais e pesquise por hemocentros do seu estado.")
    ufs = list(UF_CENTER.keys())
    base_link = "https://www.google.com/search?q=doar+sangue+{UF}+hemocentro"
    rows = []
    for uf in ufs:
        link = base_link.format(UF=uf)
        md = f"[Abrir]({link})"
        rows.append({"UF": uf, "Link": md})
    df_links = pd.DataFrame(rows)
    st.dataframe(df_links, use_container_width=True)


# -----------------------------------------------------------------------------
# Seção: Cadastro de doador (simples, local)
# -----------------------------------------------------------------------------
def pagina_cadastro():
    st.header("Cadastrar doador (opcional)")

    with st.form("form_doador", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        nome = c1.text_input("Nome completo")
        tel = c2.text_input("Telefone/WhatsApp")
        uf = c3.selectbox("UF", sorted(UF_CENTER.keys()))

        c4, c5, c6 = st.columns([2, 2, 1])
        email = c4.text_input("E-mail")
        cidade = c5.text_input("Cidade")
        tipo = c6.selectbox("Tipo sanguíneo", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])

        consent = st.checkbox("Autorizo o uso desses dados para contato sobre doação.")
        enviado = st.form_submit_button("Salvar cadastro", type="primary")

    if enviado:
        if not (nome and email and consent):
            st.warning("Preencha **Nome**, **E-mail** e marque o consentimento.")
        else:
            # No futuro: salvar em planilha/DB. Aqui mostramos um recibo simples.
            st.success("Cadastro salvo localmente (exemplo).")
            st.json({"nome": nome, "email": email, "telefone": tel, "uf": uf, "cidade": cidade, "tipo": tipo})


# -----------------------------------------------------------------------------
# Seção: Sobre
# -----------------------------------------------------------------------------
def pagina_sobre():
    st.header("Sobre este painel")
    st.markdown(
        textwrap.dedent(
            """
            **VisioData** — fontes oficiais e dados agregados, prontos para apresentação.
            
            - **ANVISA (nacional)**: leia o CSV público do projeto Hemoprod, gere KPIs e mapa por UF.
            - **Estoques estaduais**: atalhos para pesquisa de hemocentros por estado.
            - **Cadastrar doador**: formulário simples (exemplo local) para futuras integrações.
            
            **Observação:** alguns CSVs da ANVISA variam de separador e codificação.  
            O carregamento “robusto” tenta múltiplas estratégias (detecção de separador, `on_bad_lines='skip'`, etc.).
            """
        )
    )


# -----------------------------------------------------------------------------
# Layout principal – barra lateral e roteamento
# -----------------------------------------------------------------------------
st.sidebar.subheader("Navegação")
secao = st.sidebar.radio(
    label="Navegação",
    options=["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"],
    index=0,
    label_visibility="collapsed",
)

st.sidebar.markdown(
    """
    <div style="
      margin-top:1rem;
      font-size:.9rem;
      background:#fff7ed;border:1px solid #fed7aa;
      padding:.7rem;border-radius:.4rem;">
      💡 <b>Dica:</b> use o botão <b>Carregar agora</b> para atualizar
      diretamente do site da Anvisa.
    </div>
    """,
    unsafe_allow_html=True,
)

if secao == "ANVISA (nacional)":
    pagina_anvisa()
elif secao == "Estoques estaduais":
    pagina_links_estaduais()
elif secao == "Cadastrar doador":
    pagina_cadastro()
else:
    pagina_sobre()
