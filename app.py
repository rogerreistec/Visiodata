# app.py
# VisioData – Painel de Estoques e Produção Hemoterápica

from __future__ import annotations

import io
import unicodedata
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk


# -----------------------------------------------------------------------------
# Configuração da página
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VisioData | Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
)

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
    """Lê CSV (URL ou upload) sem usar low_memory+python, com tentativas de sep."""
    if uploaded:
        byts = origem if isinstance(origem, (bytes, bytearray)) else origem.read()
        buf = io.BytesIO(byts)
    else:
        buf = origem  # URL string

    # 1) autodetect
    try:
        df = pd.read_csv(buf, sep=None, engine="python", on_bad_lines="skip", dtype=str)
        if df.empty:
            raise ValueError("CSV vazio.")
        return df
    except Exception:
        pass

    # 2) força ';'
    try:
        if uploaded:
            buf.seek(0)
        df = pd.read_csv(buf, sep=";", engine="python", on_bad_lines="skip", dtype=str)
        if df.empty:
            raise ValueError("CSV vazio.")
        return df
    except Exception:
        pass

    # 3) força ','
    if uploaded:
        buf.seek(0)
    df = pd.read_csv(buf, sep=",", engine="python", on_bad_lines="skip", dtype=str)
    if df.empty:
        raise ValueError("CSV vazio.")
    return df


def normaliza_colunas(df: pd.DataFrame) -> pd.DataFrame:
    ren = {c: " ".join(c.strip().lower().split()) for c in df.columns}
    df = df.rename(columns=ren)
    drop_cols = [c for c in df.columns if c.startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")
    return df


def detecta_colunas(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    # ano
    col_ano = None
    for c in df.columns:
        if "ano" in c and ("ref" in c or "refer" in c or "referência" in c):
            col_ano = c
            break

    # uf
    col_uf = None
    for c in df.columns:
        if c == "uf" or c.endswith(" uf") or c.startswith("uf "):
            col_uf = c
            break

    # métricas possíveis
    possiveis = []
    for c in df.columns:
        if c in {col_ano, col_uf}:
            continue
        amostra = pd.to_numeric(
            df[c].dropna().astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )
        if amostra.notna().sum() > 0:
            possiveis.append(c)
    if "id da resposta" in df.columns and "id da resposta" in possiveis:
        possiveis = ["id da resposta"] + [x for x in possiveis if x != "id da resposta"]
    return col_ano, col_uf, possiveis


def to_numeric_safe(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
        errors="coerce",
    )


# centroides aproximados por UF (para o mapa)
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

# para normalizar nomes de estados -> siglas
UF_NOMES = {
    "ACRE": "AC", "ALAGOAS": "AL", "AMAPA": "AP", "AMAPÁ": "AP", "AMAZONAS": "AM",
    "BAHIA": "BA", "CEARA": "CE", "CEARÁ": "CE", "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES", "ESPÍRITO SANTO": "ES", "GOIAS": "GO", "GOIÁS": "GO",
    "MARANHAO": "MA", "MARANHÃO": "MA", "MATO GROSSO DO SUL": "MS",
    "MATO GROSSO": "MT", "MINAS GERAIS": "MG", "PARA": "PA", "PARÁ": "PA",
    "PARAIBA": "PB", "PARAÍBA": "PB", "PARANA": "PR", "PARANÁ": "PR",
    "PERNAMBUCO": "PE", "PIAUI": "PI", "PIAUÍ": "PI", "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN", "RIO GRANDE DO SUL": "RS", "RONDONIA": "RO",
    "RONDÔNIA": "RO", "RORAIMA": "RR", "SANTA CATARINA": "SC",
    "SAO PAULO": "SP", "SÃO PAULO": "SP", "SERGIPE": "SE", "TOCANTINS": "TO"
}


def strip_accents_upper(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "").encode("ascii", "ignore").decode("ascii")
    return " ".join(s.upper().split())


def uf_para_sigla(valor: str) -> str | None:
    if valor is None or str(valor).strip() == "":
        return None
    v = str(valor).strip()
    if len(v) <= 2:
        return v.upper()
    v2 = strip_accents_upper(v)
    return UF_NOMES.get(v2)


def kpi_box(label: str, value: int | float | str):
    st.metric(label, value)


# -----------------------------------------------------------------------------
# Seção: ANVISA (nacional)  —  **AJUSTADA (persistência + filtro + mapa)**
# -----------------------------------------------------------------------------
def pagina_anvisa():
    st.header("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    # URL / Upload
    with st.expander("URL do CSV (Hemoprod — ANVISA)", expanded=True):
        url_default = (
            "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
            "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
            "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
        )
        url = st.text_input("Cole a URL do CSV aqui", url_default, key="hemoprod_url")

        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            carregar = st.button("Carregar agora", type="primary")
        with c2:
            up = st.file_uploader("…ou envie o CSV", type=["csv"])
        with c3:
            limpar = st.button("Limpar base")

    # reseta base
    if limpar:
        st.session_state.pop("hemoprod_df", None)

    # lê/guarda base
    if carregar or (up is not None):
        try:
            if up is not None:
                df = read_csv_robusto(up.getvalue(), uploaded=True)
            else:
                df = read_csv_robusto(url, uploaded=False)
            df = normaliza_colunas(df)
            st.session_state["hemoprod_df"] = df
            st.success("Base carregada com sucesso!")
        except Exception as e:
            st.error(f"Falha ao carregar: {e}")
            return

    # recupera base persistida
    df = st.session_state.get("hemoprod_df")
    if df is None:
        st.info("Cole a URL ou envie o CSV e clique **Carregar agora**.")
        return

    # detecta colunas e prepara controles
    col_ano, col_uf, metricas = detecta_colunas(df)

    # controles (com chaves -> persistem em cada rerun)
    cc1, cc2, cc3, cc4 = st.columns([1.2, 1.2, 1.2, 1])

    # Ano
    with cc1:
        anos_opc = ["(Todos)"]
        ano_recente = None
        if col_ano and df[col_ano].notna().any():
            anos_num = pd.to_numeric(df[col_ano], errors="coerce")
            if anos_num.notna().any():
                anos_unicos = sorted(list(set(anos_num.dropna().astype(int).tolist())), reverse=True)
                ano_recente = int(anos_num.dropna().max())
                anos_opc = ["(Todos)", "(Mais recente)"] + anos_unicos
        ano_escolhido = st.selectbox("Ano", anos_opc, key="anv_ano_escolhido")

    # Coluna UF
    with cc2:
        # preseleciona a coluna detectada
        opcoes_uf = ["<não há>"] + list(df.columns)
        default_uf = (df.columns.tolist().index(col_uf) + 1) if col_uf in df.columns else 0
        uf_col = st.selectbox("Coluna UF (se existe)", opcoes_uf, index=default_uf, key="anv_uf_col")
        if uf_col == "<não há>":
            uf_col = None

    # Métrica
    with cc3:
        if len(metricas) == 0:
            metricas = [c for c in df.columns if c not in {col_ano, uf_col}]
        met_col = st.selectbox("Coluna MÉTRICA (para Soma)", metricas, key="anv_metrica")

    # Agregação
    with cc4:
        oper = st.selectbox("Agregação", ["Soma", "Contagem"], index=0, key="anv_oper")

    # ----------------- FILTRO POR ANO -----------------
    df_ag = df.copy()
    if col_ano and ano_escolhido != "(Todos)":
        if ano_escolhido == "(Mais recente)" and (ano_recente is not None):
            df_ag = df_ag[df_ag[col_ano].astype(str) == str(ano_recente)]
        elif ano_escolhido not in {"(Todos)", "(Mais recente)"}:
            df_ag = df_ag[df_ag[col_ano].astype(str) == str(ano_escolhido)]

    # ----------------- KPI -----------------
    total_reg = len(df_ag)
    anos_dist = df_ag[col_ano].nunique(dropna=True) if col_ano else 0
    ufs_dist = df_ag[uf_col].nunique(dropna=True) if uf_col else 0

    # ----------------- AGREGAÇÃO POR UF -----------------
    if uf_col:
        # normaliza UF (suporta 'SP' ou 'São Paulo')
        uf_norm = df_ag[uf_col].astype(str).map(uf_para_sigla)
        # onde ainda None, mantém uppercase original (para quem já era sigla)
        uf_norm = uf_norm.fillna(df_ag[uf_col].astype(str).str.upper().str.strip())
        df_ag = df_ag.assign(__uf__=uf_norm)

        df_ag["__valor__"] = 1.0
        if oper == "Soma":
            df_ag["__valor__"] = to_numeric_safe(df_ag[met_col])

        grupo = (
            df_ag.groupby("__uf__", dropna=False, as_index=False)["__valor__"]
            .sum()
            .rename(columns={"__uf__": "uf", "__valor__": "valor"})
        )
        grupo["uf"] = grupo["uf"].astype(str).str.upper().str.strip()
        grupo = grupo[grupo["uf"].isin(UF_CENTER.keys())]
    else:
        grupo = pd.DataFrame(columns=["uf", "valor"])

    total_agregado = float(grupo["valor"].sum()) if len(grupo) else 0.0

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_box("Registros", f"{total_reg:,}".replace(",", "."))
    with k2:
        kpi_box("Anos distintos", int(anos_dist))
    with k3:
        kpi_box("UF distintas", int(ufs_dist))
    with k4:
        kpi_box(("Total (Soma)" if oper == "Soma" else "Total (Contagem)"),
                f"{int(total_agregado):,}".replace(",", ".") if total_agregado.is_integer()
                else f"{total_agregado:,.2f}".replace(",", "."))

    # ----------------- MAPA -----------------
    st.subheader("Mapa por UF")
    if len(grupo) == 0:
        st.info("Não há dados suficientes para o mapa (verifique a coluna UF e a agregação).")
    else:
        plot_df = []
        vmax = float(grupo["valor"].max() or 1.0)
        for _, r in grupo.iterrows():
            uf = r["uf"]
            val = float(r["valor"]) if pd.notna(r["valor"]) else 0.0
            if uf in UF_CENTER:
                lat, lon = UF_CENTER[uf]
                plot_df.append(dict(uf=uf, valor=val, lat=lat, lon=lon))
        plot_df = pd.DataFrame(plot_df)
        if not plot_df.empty:
            plot_df["radius"] = 6000 + 4000 * np.sqrt(plot_df["valor"] / (vmax if vmax else 1))
            layer = pdk.Layer(
                "ScatterplotLayer",
                data=plot_df,
                get_position=["lon", "lat"],
                get_radius="radius",
                get_fill_color=[244, 63, 94, 170],
                pickable=True,
            )
            view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5)
            tooltip = {"text": "{uf}: {valor}"}
            st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state,
                                     tooltip=tooltip, map_style="light"),
                            use_container_width=True)
        else:
            st.info("Sem pontos válidos para plotar (verifique as UFs).")

    # ----------------- TABELA -----------------
    st.subheader("Tabela agregada por UF")
    st.dataframe(grupo.sort_values("valor", ascending=False), use_container_width=True)


# -----------------------------------------------------------------------------
# Seções que já estavam OK (mantidas)
# -----------------------------------------------------------------------------
def pagina_links_estaduais():
    st.header("Acesse páginas/oficiais e pesquise por hemocentros do seu estado.")
    ufs = list(UF_CENTER.keys())
    base_link = "https://www.google.com/search?q=doar+sangue+{UF}+hemocentro"
    df_links = pd.DataFrame({"UF": ufs, "Link": [f"[Abrir]({base_link.format(UF=u)})" for u in ufs]})
    st.dataframe(df_links, use_container_width=True)


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
            st.success("Cadastro salvo localmente (exemplo).")
            st.json({"nome": nome, "email": email, "telefone": tel, "uf": uf, "cidade": cidade, "tipo": tipo})


def pagina_sobre():
    st.header("Sobre este painel")
    st.markdown(
        """
        **VisioData** — fontes oficiais e dados agregados, prontos para apresentação.
        - **ANVISA (nacional)**: leia o CSV público do Hemoprod, gere KPIs e mapa por UF.
        - **Estoques estaduais**: atalhos para pesquisa de hemocentros por estado.
        - **Cadastrar doador**: formulário simples (exemplo local).
        """
    )


# -----------------------------------------------------------------------------
# Navegação (sidebar)
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
      margin-top:1rem;font-size:.9rem;background:#fff7ed;border:1px solid #fed7aa;
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
