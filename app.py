# app.py
# VisioData – Painel de Estoques e Produção Hemoterápica (ANVISA/Hemoprod)
# -----------------------------------------------------------------------
# O foco deste app é ser simples para quem vê e robusto para quem exige:
#  - barra lateral fixa com logo
#  - downloader do CSV da Anvisa com tratamento de erros
#  - KPIs automáticos e Mapa por UF (com chaves únicas para evitar conflitos)
#  - lista de links para todos os estados
#  - formulário opcional de cadastro de doador
# -----------------------------------------------------------------------

from __future__ import annotations

import io
import sys
import time
import textwrap
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st


# ==========================================================
# ---------------------- CONFIG GERAL ----------------------
# ==========================================================
st.set_page_config(
    page_title="VisioData | Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

# ======== caminhos/arquivos esperados
LOGO_PATH_SVG = "ativos/logo.svg"
LOGO_PATH_PNG = "ativos/logo.png"

# ======== URL padrão da Anvisa (Hemoprod – nacional)
URL_PADRAO_ANVISA = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

# ======== centroides simples por UF (latitude/longitude)
UF_CENTROIDES: Dict[str, Tuple[float, float]] = {
    # lat, lon (aproximados)
    "AC": (-9.02, -70.81),
    "AL": (-9.62, -36.82),
    "AM": (-3.07, -60.02),
    "AP": (0.04, -51.06),
    "BA": (-12.97, -38.51),
    "CE": (-3.72, -38.54),
    "DF": (-15.78, -47.93),
    "ES": (-20.32, -40.34),
    "GO": (-16.68, -49.25),
    "MA": (-2.53, -44.30),
    "MG": (-19.92, -43.94),
    "MS": (-20.45, -54.62),
    "MT": (-15.60, -56.10),
    "PA": (-1.45, -48.49),
    "PB": (-7.12, -34.86),
    "PE": (-8.05, -34.90),
    "PI": (-5.09, -42.80),
    "PR": (-25.43, -49.27),
    "RJ": (-22.91, -43.20),
    "RN": (-5.81, -35.21),
    "RO": (-8.76, -63.90),
    "RR": (2.82, -60.67),
    "RS": (-30.03, -51.23),
    "SC": (-27.59, -48.55),
    "SE": (-10.91, -37.07),
    "SP": (-23.55, -46.64),
    "TO": (-10.25, -48.33),
}

# ======== links rápidos (usamos busca do Google para evitar links quebrados)
LINKS_POR_UF = {
    uf: f"https://www.google.com/search?q=doar+sangue+{uf}+hemocentro"
    for uf in UF_CENTROIDES.keys()
}


# ==========================================================
# --------------------- FUNÇÕES AUX ------------------------
# ==========================================================
def sidebar_logo():
    """Exibe a logomarca na barra lateral, no topo."""
    st.sidebar.markdown("### 🩸 VisioData")
    # tenta SVG depois PNG
    try:
        st.sidebar.image(LOGO_PATH_SVG, use_column_width=True)
    except Exception:
        try:
            st.sidebar.image(LOGO_PATH_PNG, use_column_width=True)
        except Exception:
            st.sidebar.info("Envie sua logo em `ativos/logo.svg` (ou `logo.png`).")


@st.cache_data(show_spinner=False, ttl=60 * 30)
def baixar_csv(url: str) -> str:
    """
    Faz download do CSV remoto e devolve o conteúdo bruto (texto).
    Cacheado por 30 min para aliviar a Anvisa.
    """
    import requests  # import local para evitar overhead no startup

    # Alguns servidores exigem User-Agent legível
    headers = {
        "User-Agent": "VisioData/1.0 (+https://github.com/rogerreistec/Visiodata)"
    }

    # Tentativas com backoff simples
    ultimo_erro: Optional[str] = None
    for i in range(3):
        try:
            resp = requests.get(url, headers=headers, timeout=60)
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as e:
            ultimo_erro = str(e)
            time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"Falha ao baixar o CSV:\n{ultimo_erro}")


def ler_csv_robusto(conteudo: str) -> pd.DataFrame:
    """
    Lê o CSV da Anvisa tolerando variações (separador, encoding, decimal, etc.).
    Tenta ; e , como separadores. Faz coerção de números para float quando possível.
    """
    # 1) tenta separador ';'
    for sep in [";", ",", "\t", "|"]:
        try:
            df = pd.read_csv(
                io.StringIO(conteudo),
                sep=sep,
                engine="python",
                low_memory=False,
            )
            if df.shape[1] >= 2:
                return df
        except Exception:
            pass
    # fallback “esperto”: tenta auto
    df = pd.read_csv(io.StringIO(conteudo), engine="python", low_memory=False)
    return df


def normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza nomes de colunas (minúsculas, sem acentos simples e com underscore).
    Não muda o conteúdo.
    """
    def _limpa(s: str) -> str:
        s2 = (
            s.lower()
            .replace(" ", "_")
            .replace("-", "_")
            .replace(".", "_")
            .replace("/", "_")
        )
        # mapeamento mínimo de acentos comuns
        m = str.maketrans("áàâãéêíóôõúç", "aaaaeeiooouc")
        return s2.translate(m)

    df = df.copy()
    df.columns = [_limpa(c) for c in df.columns]
    return df


def escolhe_coluna_existente(df: pd.DataFrame, candidatos: list[str]) -> Optional[str]:
    """Retorna a primeira coluna existente em df, dentre as candidatas."""
    for c in candidatos:
        if c in df.columns:
            return c
    return None


def agrega_por_uf(
    df: pd.DataFrame,
    col_ano: str,
    col_uf: str,
    col_metrica: str,
    func: str = "soma",
) -> pd.DataFrame:
    """
    Gera uma tabela agregada por UF (para ano e métrica selecionados).
    """
    d = df.copy()

    # Coerção simples para números
    d[col_metrica] = pd.to_numeric(d[col_metrica], errors="coerce")

    # Mantém somente linhas com UF existente no dicionário e ano escolhido
    d = d[d[col_uf].isin(UF_CENTROIDES.keys())]

    # Seleciona agregador
    agg = {"soma": "sum", "média": "mean", "contagem": "count"}.get(func, "sum")

    out = d.groupby([col_ano, col_uf], dropna=False)[col_metrica].agg(agg).reset_index()
    out.rename(columns={col_metrica: "valor"}, inplace=True)
    return out


def mapa_por_uf(df_agg_ano: pd.DataFrame, ano: str) -> pdk.Deck:
    """
    Cria um mapa (pydeck) de bolhas proporcionais ao valor por UF.
    """
    if df_agg_ano.empty:
        # retorno vazio – o chamador decide o que fazer
        return pdk.Deck()

    dados = []
    vmax = float(df_agg_ano["valor"].max()) if not df_agg_ano["valor"].empty else 0.0

    for _, row in df_agg_ano.iterrows():
        uf = row["uf"]
        valor = float(row["valor"]) if pd.notna(row["valor"]) else 0.0
        lat, lon = UF_CENTROIDES.get(uf, (None, None))
        if lat is None:
            continue
        # tamanho da bolha proporcional, com mínimo p/ aparecer
        radius = 1_000 + (0 if vmax <= 0 else int(12_000 * (valor / vmax)))
        dados.append({"position": [lon, lat], "uf": uf, "valor": valor, "radius": radius})

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=dados,
        get_position="position",
        get_radius="radius",
        get_fill_color="[255, 64, 64, 160]",
        pickable=True,
        auto_highlight=True,
    )

    view_state = pdk.ViewState(latitude=-15.7, longitude=-52.0, zoom=3.3)
    tooltip = {"html": "<b>UF: </b>{uf}<br/><b>Valor:</b> {valor}", "style": {"color": "white"}}

    deck = pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip)
    return deck


# ==========================================================
# --------------------- INTERFACE UI -----------------------
# ==========================================================
sidebar_logo()

st.sidebar.caption(
    "Fontes oficiais e dados agregados — prontos para apresentação acadêmica."
)

secao = st.sidebar.radio(
    "Navegação",
    options=["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "💡 Dica: use o botão **Carregar agora** para atualizar diretamente do site da Anvisa."
)


# ==========================================================
# ---------------- SEÇÃO: ANVISA (nacional) ---------------
# ==========================================================
if secao == "ANVISA (nacional)":
    st.title("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    with st.expander("URL do CSV (Hemoprod — ANVISA)", expanded=True):
        url = st.text_input(
            "Cole a URL do CSV da Anvisa",
            value=URL_PADRAO_ANVISA,
            key="url_anvisa",
            label_visibility="collapsed",
        )

        c1, c2 = st.columns([1, 3])
        with c1:
            btn = st.button("Carregar agora", use_container_width=True)
        with c2:
            st.caption("…ou envie o CSV (alternativa)")

        arquivo = st.file_uploader(
            "Envio opcional",
            type=["csv"],
            key="up_anvisa",
            label_visibility="collapsed",
        )

    # ----- Leitura de dados
    df: Optional[pd.DataFrame] = None
    erro = None

    if btn and not arquivo:
        try:
            bruto = baixar_csv(url.strip())
            df = ler_csv_robusto(bruto)
        except Exception as e:
            erro = f"Falha ao carregar: {e}"

    if arquivo is not None:
        try:
            conteudo = arquivo.read().decode("utf-8", errors="ignore")
            df = ler_csv_robusto(conteudo)
        except Exception as e:
            erro = f"Falha ao ler o arquivo enviado: {e}"

    # Caso o usuário tenha acabado de entrar (sem apertar nada), tentamos
    # mostrar algo default (pode demorar – por isso, deixamos ao clique).
    if df is None and erro is None and not btn and arquivo is None:
        st.info("Use **Carregar agora** para buscar os dados oficiais.")

    if erro:
        st.error(erro)

    if df is not None:
        df = normaliza_colunas(df)

        st.markdown("### KPIs automáticos")
        # heurística para achar colunas comuns
        col_ano = escolhe_coluna_existente(df, ["ano_de_referencia", "ano", "ano_referencia"]) or \
                  st.selectbox("Coluna de ano (se não detectar)", df.columns, key="kpi_ano_fallback")

        col_uf = escolhe_coluna_existente(df, ["uf", "estado", "sigla_uf"]) or \
                 st.selectbox("Coluna UF (se existir)", df.columns, key="kpi_uf_fallback")

        # métrica: oferecemos todas numéricas + fallback para qualquer
        candidatos_metricos = [c for c in df.columns if pd.to_numeric(df[c], errors="coerce").notna().any()]
        col_metrica = st.selectbox(
            "Coluna MÉTRICA (numérica)",
            candidatos_metricos if candidatos_metricos else list(df.columns),
            key="kpi_metrica",
        )

        # Indicadores de quantidades
        st.write("Registros:", f"**{len(df):,}**".replace(",", "."))
        if col_ano in df.columns:
            st.write("Anos distintos:", f"**{df[col_ano].nunique()}**")
        if col_uf in df.columns:
            st.write("UF distintas:", f"**{df[col_uf].nunique()}**")

        st.divider()

        # --- Agregações e mapa
        anos_disponiveis = (
            sorted([str(x) for x in df[col_ano].dropna().unique()])
            if col_ano in df.columns
            else []
        )
        ano_escolhido = st.selectbox(
            "Selecione o ano",
            options=anos_disponiveis,
            index=(anos_disponiveis.index(max(anos_disponiveis)) if anos_disponiveis else 0),
            key="map_ano",
        )

        func_agg = st.selectbox(
            "Agregação",
            options=["soma", "média", "contagem"],
            index=0,
            key="map_agg",
        )

        # Agrega (tabela por [ano, uf] -> valor)
        df_agg = agrega_por_uf(df, col_ano, col_uf, col_metrica, func=func_agg)
        df_ano = df_agg[df_agg[col_ano].astype(str) == str(ano_escolhido)].copy()

        # Mapa
        st.subheader("Mapa por UF")
        deck = mapa_por_uf(df_ano[[col_ano, "uf", "valor"]], ano_escolhido)
        if not deck.layers:
            st.info("Sem dados suficientes para o mapa (verifique UF e métrica).")
        else:
            st.pydeck_chart(deck, use_container_width=True)

        # Links rápidos (todos os estados)
        st.subheader("Links rápidos por UF")
        tabela_links = pd.DataFrame(
            {
                "UF": list(UF_CENTROIDES.keys()),
                "Valor": [df_ano.loc[df_ano["uf"] == uf, "valor"].sum() if not df_ano.empty else np.nan
                          for uf in UF_CENTROIDES.keys()],
                "Link": [LINKS_POR_UF[uf] for uf in UF_CENTROIDES.keys()],
            }
        )
        # Render com col de link clicável
        st.dataframe(
            tabela_links.style.format({"Link": lambda x: f'=HYPERLINK("{x}", "Abrir")'}),
            use_container_width=True,
            height=320,
        )

        st.divider()

        # Tabela detalhada (amostra)
        st.subheader("Amostra dos dados")
        st.dataframe(df.head(200), use_container_width=True, height=360)


# ==========================================================
# --------------- SEÇÃO: ESTOQUES ESTADUAIS ---------------
# (esta é uma área estática p/ links úteis por UF)
# ==========================================================
elif secao == "Estoques estaduais":
    st.title("Links úteis por UF – Estoques/Doação")
    st.caption("Acesse páginas/oficiais e pesquise por hemocentros do seu estado.")

    df_links = pd.DataFrame(
        {"UF": list(LINKS_POR_UF.keys()), "Link": list(LINKS_POR_UF.values())}
    )
    st.dataframe(
        df_links.style.format({"Link": lambda x: f'=HYPERLINK("{x}", "Abrir")'}),
        use_container_width=True,
        height=550,
    )


# ==========================================================
# --------------- SEÇÃO: CADASTRAR DOADOR -----------------
# ==========================================================
elif secao == "Cadastrar doador":
    st.title("🧑‍⚕️ Cadastrar doador (opcional)")

    with st.form("form_doador", clear_on_submit=True):
        c1, c2, c3 = st.columns([2, 2, 1])
        nome = c1.text_input("Nome completo", key="doa_nome")
        tel  = c2.text_input("Telefone/WhatsApp", key="doa_tel")
        uf   = c3.selectbox("UF", options=list(UF_CENTROIDES.keys()), key="doa_uf", index=25)

        c4, c5, c6 = st.columns([2, 2, 1])
        email = c4.text_input("E-mail", key="doa_email")
        cidade = c5.text_input("Cidade", key="doa_cidade")
        tipo   = c6.selectbox("Tipo sanguíneo", options=["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], key="doa_tipo")

        consent = st.checkbox(
            "Autorizo o uso desses dados para contato sobre doação.",
            key="doa_ok",
        )
        enviar = st.form_submit_button("Salvar cadastro")

        if enviar:
            if not (nome and email and consent):
                st.error("Preencha ao menos **Nome**, **E-mail** e marque o consentimento.")
            else:
                # Aqui você integraria com um backend/planilha/DB
                st.success(
                    f"Cadastro salvo! Obrigado, **{nome}**. "
                    f"UF **{uf}**, tipo **{tipo}**. Entraremos em contato por **{email}**."
                )


# ==========================================================
# ---------------------- SEÇÃO: SOBRE ----------------------
# ==========================================================
else:
    st.title("Sobre este painel")
    st.markdown(
        textwrap.dedent(
            """
            **VisioData** é um painel didático de dados hemoterápicos:
            - Fontes oficiais (ANVISA/Hemoprod).
            - KPIs automáticos e mapa por UF.
            - Links úteis por estado.
            - Cadastro opcional de doadores.

            **Como usar**
            1. Vá em *ANVISA (nacional)* e clique em **Carregar agora** para buscar o CSV.
            2. Ajuste *Ano*, *UF* e *Métrica*.
            3. Use o *Mapa por UF* e *Links rápidos* para explorar.

            **Observações técnicas**
            - Widgets possuem *keys* únicas para evitar o erro `StreamlitDuplicateElementId`.
            - Leitura de CSV tolerante a variações de separador/encoding.
            - O mapa usa **pydeck** com centróides de cada UF e bolhas proporcionais ao valor.
            - Os dados carregados ficam em *cache* por 30 min (reduz tráfego na Anvisa).
            """
        )
    )

