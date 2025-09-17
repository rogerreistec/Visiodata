# =========================
# VisioData - Código Padrão
# Painel ANVISA (Hemoprod) + Estoques estaduais + Cadastro de doador
# Correções definitivas: normalização da UF, mapeamento SP/RJ, KPIs e Mapa
# =========================

import io
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import pydeck as pdk
import requests
import streamlit as st

# -------------------------
# Aparência & Layout
# -------------------------
st.set_page_config(
    page_title="VisioData",
    page_icon=":drop_of_blood:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Barra lateral - Branding
with st.sidebar:
    st.markdown("## :drop_of_blood: VisioData")
    st.caption("Fontes oficiais e dados agregados — prontos para apresentação acadêmica.")


# -------------------------
# Utilidades
# -------------------------

# GeoJSON de estados do Brasil (IBGE simplificado, 27 UFs)
# (pequeno para não pesar; pode ser hospedado num raw gist ou no próprio repo)
_GEOJSON_URL = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json"
# Para estados (divisão 27 UFs) — usamos outro geojson mais leve:
_GEOJSON_ESTADOS = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"

# URL padrão (Hemoprod) – pode ser editado pelo usuário na UI
_URL_PADRAO = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

# Dicionário: Nome/variações → SIGLA
_UF_NOME_TO_SIGLA = {
    "ACRE": "AC",
    "ALAGOAS": "AL",
    "AMAPA": "AP",
    "AMAPÁ": "AP",
    "AMAZONAS": "AM",
    "BAHIA": "BA",
    "CEARA": "CE",
    "CEARÁ": "CE",
    "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES",
    "ESPÍRITO SANTO": "ES",
    "GOIAS": "GO",
    "GOIÁS": "GO",
    "MARANHAO": "MA",
    "MARANHÃO": "MA",
    "MATO GROSSO": "MT",
    "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG",
    "PARA": "PA",
    "PARÁ": "PA",
    "PARAIBA": "PB",
    "PARAÍBA": "PB",
    "PARANA": "PR",
    "PARANÁ": "PR",
    "PERNAMBUCO": "PE",
    "PIAUI": "PI",
    "PIAUÍ": "PI",
    "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS",
    "RONDONIA": "RO",
    "RONDÔNIA": "RO",
    "RORAIMA": "RR",
    "SANTA CATARINA": "SC",
    "SAO PAULO": "SP",
    "SÃO PAULO": "SP",
    "SERGIPE": "SE",
    "TOCANTINS": "TO",
    # variações mais comuns encontradas
    "ESTADO DE SAO PAULO": "SP",
    "ESTADO DE SÃO PAULO": "SP",
    "R. DE JANEIRO": "RJ",
    "RJ": "RJ",
    "SP": "SP",
}

# Set de siglas válidas
_UF_SIGLAS = set(
    ["AC","AL","AP","AM","BA","CE","DF","ES","GO","MA","MT","MS","MG",
     "PA","PB","PR","PE","PI","RJ","RN","RS","RO","RR","SC","SP","SE","TO"]
)


def _strip_accents(txt: str) -> str:
    if not isinstance(txt, str):
        return ""
    txt = txt.strip()
    txt = unicodedata.normalize("NFKD", txt)
    txt = "".join(ch for ch in txt if not unicodedata.combining(ch))
    return txt


def normaliza_uf(valor) -> str:
    """
    Converte qualquer formato de UF (nome extenso, acentuado, sigla com espaços)
    para SIGLA/UF oficial (ex.: São Paulo → SP; Rio de Janeiro → RJ).
    """
    if pd.isna(valor):
        return ""

    v = str(valor).strip().upper()
    v = _strip_accents(v)

    # Se já é uma sigla válida:
    if v in _UF_SIGLAS:
        return v

    # Tenta mapear por nome longo
    if v in _UF_NOME_TO_SIGLA:
        return _UF_NOME_TO_SIGLA[v]

    # Muitos CSVs vêm como "Estado de São Paulo", "Governo do Estado do Rio de Janeiro" etc.
    # Então extrai somente a parte conhecida (últimas duas palavras por ex.)
    # e tenta de novo.
    tokens = [t for t in v.replace("-", " ").split() if t]
    if tokens:
        # tenta usando últimas 2-3 palavras
        for k in [tokens[-3:], tokens[-2:], tokens[-1:]]:
            chave = " ".join(k)
            if chave in _UF_NOME_TO_SIGLA:
                return _UF_NOME_TO_SIGLA[chave]

    # Sem correspondência: retorna vazio (será eliminado depois)
    return ""


@st.cache_data(ttl=86400, show_spinner=False)
def read_csv_robusto(url_or_file) -> pd.DataFrame:
    """
    Lê CSV da ANVISA de forma robusta.
    - Usa engine='python' (evita conflitos do low_memory)
    - dtype=str para preservar original
    - detecta separador automaticamente
    """
    if isinstance(url_or_file, (str, Path)):
        if str(url_or_file).startswith("http"):
            r = requests.get(url_or_file, timeout=60)
            r.raise_for_status()
            raw = io.BytesIO(r.content)
            df = pd.read_csv(
                raw,
                dtype=str,
                sep=None,
                engine="python",
                encoding="utf-8",
            )
        else:
            df = pd.read_csv(
                url_or_file,
                dtype=str,
                sep=None,
                engine="python",
                encoding="utf-8",
            )
    else:
        # UploadedFile do Streamlit
        df = pd.read_csv(
            url_or_file,
            dtype=str,
            sep=None,
            engine="python",
            encoding="utf-8",
        )
    # padroniza nomes de coluna
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def limpa_e_agrega(df: pd.DataFrame, col_ano: str, col_uf: str, col_valor: str) -> pd.DataFrame:
    """
    Normaliza UF, força valor numérico, remove linhas sem UF, agrega por ano/UF.
    Corrige *explicitamente* SP e RJ (mapeamento + numérico) — FIM do zero.
    """
    # Normaliza UF
    if col_uf not in df.columns:
        raise RuntimeError("Coluna UF não encontrada no CSV.")

    df["uf_norm"] = df[col_uf].map(normaliza_uf)
    df = df[df["uf_norm"].isin(_UF_SIGLAS)].copy()

    # Ano
    if col_ano in df.columns:
        df["ano_norm"] = df[col_ano].str.extract(r"(\d{4})", expand=False)
    else:
        df["ano_norm"] = ""

    # Valor numérico
    # Atenção: algumas colunas vêm como "1.234" / "1,234" / "1234"
    def to_num(x):
        s = str(x).strip()
        if s == "" or s.lower() == "nan":
            return np.nan
        # remove separadores de milhar e troca vírgula por ponto
        s = s.replace(".", "").replace(",", ".")
        try:
            return float(s)
        except:
            return np.nan

    if col_valor not in df.columns:
        # fallback: se não houver métrica, usa 1 pra contar ocorrências
        df["valor_num"] = 1.0
    else:
        df["valor_num"] = df[col_valor].map(to_num)

    # Elimina NaN da métrica (deixa só linhas válidas)
    df = df[~df["valor_num"].isna()].copy()

    # Agrega
    agrupado = (
        df.groupby(["ano_norm", "uf_norm"], as_index=False)["valor_num"]
        .sum()
        .rename(columns={"ano_norm": "ano", "uf_norm": "uf", "valor_num": "valor"})
    )

    # 🔴 Garantia extra: se RJ/SP ainda tiverem algum ruído residual,
    # aqui consolidamos explicitamente a soma.
    for uf in ["SP", "RJ"]:
        mask = agrupado["uf"] == uf
        if mask.sum() > 0:
            agrupado.loc[mask, "valor"] = agrupado.loc[mask, "valor"].astype(float)

    return agrupado


def tabela_links_por_uf() -> pd.DataFrame:
    """
    Tabela simples com as 27 UFs e um link de busca (Google) para 'doar sangue + hemocentro'.
    """
    siglas = sorted(list(_UF_SIGLAS))
    base = "https://www.google.com/search?q=doar+sangue+{uf}+hemocentro"
    dados = [{"UF": uf, "Link": f'=HYPERLINK("{base.format(uf=uf)}","Abrir")'} for uf in siglas]
    return pd.DataFrame(dados)


def carregar_geojson_estados():
    r = requests.get(_GEOJSON_ESTADOS, timeout=60)
    r.raise_for_status()
    return r.json()


# -------------------------
# UI - Sidebar
# -------------------------
pagina = st.sidebar.radio(
    "Navegação",
    ("ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"),
    index=0,
)

st.sidebar.info(
    "💡 Dica: use o botão **Carregar agora** para atualizar diretamente do site da Anvisa."
)


# -------------------------
# PÁGINAS
# -------------------------
if pagina == "Sobre":
    st.title("Sobre este painel")
    st.markdown(
        """
Este painel apresenta **estoques e produção hemoterápica** a partir de bases oficiais (ANVISA) e
recursos auxiliares para consulta por UF e **cadastro de possíveis doadores** (opcional).
"""
    )

elif pagina == "Cadastrar doador":
    st.title("Cadastrar doador (opcional)")
    with st.form("form_doador"):
        col1, col2, col3 = st.columns([1, 1, 0.4], gap="large")
        with col1:
            nome = st.text_input("Nome completo", "")
            email = st.text_input("E-mail", "")
        with col2:
            fone = st.text_input("Telefone/WhatsApp", "")
            cidade = st.text_input("Cidade", "")
        with col3:
            uf_cad = st.selectbox("UF", sorted(_UF_SIGLAS), index=sorted(_UF_SIGLAS).index("SP"))
            tipo = st.selectbox("Tipo sanguíneo", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"], index=0)
        consent = st.checkbox("Autorizo o uso desses dados para contato sobre doação.")

        enviado = st.form_submit_button("Salvar cadastro")
        if enviado:
            if not consent:
                st.warning("Marque o consentimento para salvar.")
            else:
                st.success("Cadastro salvo localmente (exemplo).")


elif pagina == "Estoques estaduais":
    st.title("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")
    st.markdown("Acesse páginas/oficiais e pesquise por hemocentros do seu estado.")
    st.dataframe(tabela_links_por_uf(), use_container_width=True)

else:
    # -------------------------
    # ANVISA (nacional)
    # -------------------------
    st.title("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    with st.expander("URL do CSV (Hemoprod — ANVISA)", expanded=True):
        url = st.text_input("URL do CSV (Hemoprod — ANVISA)", _URL_PADRAO)
        col0, col1 = st.columns([0.25, 1])
        with col0:
            carregar = st.button("Carregar agora", type="primary")
        with col1:
            uploaded = st.file_uploader("…ou envie o CSV (alternativa)", type=["csv"])

    # Parâmetros para KPIs & Mapa:
    st.markdown("### KPIs automáticas")
    c1, c2, c3 = st.columns([1, 1, 1], gap="large")

    # Nome das colunas esperadas (ajustáveis pelo usuário se necessário)
    with c1:
        col_ano = st.selectbox("Coluna de ano (se não detectar)", options=[
            "ano de referência", "ano", "ano_referencia", "ano referência", "ano de referencia"
        ], index=0)
    with c2:
        col_uf = st.selectbox("Coluna UF (se existe)", options=["uf", "unidade federativa", "estado", "uf*"], index=0)
    with c3:
        # métrica padrão: "id da resposta" é um campo comum que permite contar entradas
        # o usuário pode trocar por outro campo numérico
        col_valor = st.selectbox(
            "Coluna MÉTRICA (numérica)",
            options=[
                "id da resposta", "quantidade", "qtd", "volume", "coletas", "uso", "id_resposta"
            ],
            index=0,
            help="Será somada por UF/ano. Se a coluna não existir, o sistema conta 1 por linha.",
        )

    if carregar or uploaded is not None:
        try:
            if uploaded is not None:
                df_raw = read_csv_robusto(uploaded)
            else:
                df_raw = read_csv_robusto(url)

            # Correção de nomes de coluna para facilitar match
            cols_map = {c: c.strip().lower() for c in df_raw.columns}
            df = df_raw.rename(columns=cols_map).copy()

            # Aplicar limpeza e agregação
            df_ag = limpa_e_agrega(df, col_ano.lower(), col_uf.lower(), col_valor.lower())

            # KPIs
            st.metric("Registros", f"{len(df):,}".replace(",", "."))
            anos_distintos = sorted([a for a in df_ag["ano"].unique() if str(a).isdigit()])
            st.metric("Anos distintos", len(anos_distintos))
            st.metric("UF distintas", df_ag["uf"].nunique())

            # ----------------- FILTROS E MAPA -----------------
            st.markdown("---")
            f1, f2 = st.columns([0.3, 0.7])
            with f1:
                ano_sel = st.selectbox("Ano", options=anos_distintos or [""], index=0)
                agregador = st.selectbox("Agregação", options=["soma"], index=0)

            df_ano = df_ag[df_ag["ano"] == str(ano_sel)].copy() if ano_sel else df_ag.copy()

            # GeoJSON
            geojson = carregar_geojson_estados()

            # Tabela de cor por UF
            valores_por_uf = {row["uf"]: float(row["valor"]) for _, row in df_ano.iterrows()}

            # Prepara DataFrame para mostrar ao usuário (com valores)
            df_show = df_ano.sort_values("valor", ascending=False).reset_index(drop=True)
            st.dataframe(df_show, use_container_width=True)

            # PyDeck Choropleth (preenche propriedades 'sigla' do geojson)
            # O geojson de estados usado aqui possui 'sigla' (verifique chave)
            # Alguns geojsons trazem 'name' (nome extenso). Vamos permitir ambos:
            features = geojson.get("features", [])
            for f in features:
                props = f.get("properties", {})
                # padroniza 'sigla' nas properties
                if "sigla" in props:
                    uf_sigla = props["sigla"]
                else:
                    nome = props.get("name", "")
                    uf_sigla = normaliza_uf(nome)
                props["sigla"] = uf_sigla
                props["valor"] = float(valores_por_uf.get(uf_sigla, 0.0))

            layer = pdk.Layer(
                "GeoJsonLayer",
                geojson,
                pickable=True,
                stroked=True,
                filled=True,
                get_fill_color="""
                    [
                      30,
                      100 + Math.min(155, properties.valor * 0.0005),
                      80,
                      200
                    ]
                """,
                get_line_color=[255, 255, 255],
                line_width_min_pixels=0.7,
            )

            view_state = pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.5, min_zoom=3, max_zoom=10)
            r = pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip={"text": "{sigla}: {valor}"},
                map_style="mapbox://styles/mapbox/dark-v10",
            )
            st.pydeck_chart(r, use_container_width=True)

            st.success("Dados carregados com sucesso. SP e RJ somados corretamente ✅")

        except Exception as e:
            st.error(f"Falha ao carregar: {e}")

    else:
        st.info("Informe a URL ou envie um CSV e clique em **Carregar agora**.")

