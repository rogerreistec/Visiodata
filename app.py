# app.py
# VisioData — Painel de Estoques e Produção Hemoterápica (ANVISA/Hemoprod)

import io
import os
import time
import json
import pandas as pd
import numpy as np
import streamlit as st
import pydeck as pdk

# ------------------------
# Configurações gerais
# ------------------------
st.set_page_config(
    page_title="VisioData | Estoques e Produção Hemoterápica",
    page_icon="🩸",
    layout="wide",
    menu_items={
        "Report a bug": "https://github.com/",
        "About": "VisioData — painel educacional para visualização de dados da ANVISA (Hemoprod)."
    },
)

# ------------------------
# Utilidades
# ------------------------

@st.cache_data(show_spinner=False)
def read_csv_robusto(entrada: str | io.BytesIO) -> pd.DataFrame:
    """
    Lê CSV com detecção automática de separador, tolerando linhas com campos a mais,
    sem usar low_memory, e tentando utf-8 -> latin-1.
    Aceita URL, caminho local, UploadedFile do Streamlit ou bytes.
    """
    kwargs_base = dict(
        engine="python",        # permite sep=None (sniff) e on_bad_lines
        sep=None,               # autodetecta separador (vírgula, ponto-e-vírgula etc.)
        on_bad_lines="skip",    # ignora linhas que não batem com o cabeçalho
        low_memory=False
    )

    # 1) Tenta utf-8
    try:
        return pd.read_csv(entrada, **kwargs_base)
    except Exception:
        pass

    # 2) Tenta latin-1
    try:
        return pd.read_csv(entrada, encoding="latin-1", **kwargs_base)
    except Exception as e:
        raise RuntimeError(f"Falha ao ler o CSV. Detalhes: {e}")

def detectar_coluna_ano(df: pd.DataFrame) -> str | None:
    candidatos = ["ano", "ano_referencia", "ano de referência", "ano de referencia"]
    for c in df.columns:
        if c.lower().strip() in candidatos:
            return c
    # fallback: alguma coluna com 4 dígitos frequentes
    for c in df.columns:
        s = df[c].astype(str).str.extract(r"(\d{4})", expand=False)
        if s.notna().mean() > 0.60:  # 60% das linhas parecem ano
            return c
    return None

def detectar_coluna_uf(df: pd.DataFrame) -> str | None:
    candidatos = ["uf", "estado", "unidade federativa", "sigla_uf"]
    for c in df.columns:
        if c.lower().strip() in candidatos:
            return c
    return None

def detectar_coluna_metrica(df: pd.DataFrame) -> str | None:
    # procura primeira numérica decente
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                return c
    return None

def make_options(current, columns):
    """Valor atual primeiro, sem duplicar."""
    cols = list(columns)
    if current and current in cols:
        return [current] + [c for c in cols if c != current]
    return cols

# Coordenadas aproximadas por UF (capitais) para o mapa (pydeck)
UF_LATLON = {
    "AC": (-9.97499, -67.8243),   "AL": (-9.66599, -35.7350),  "AM": (-3.13163, -60.0233),
    "AP": (0.033, -51.05),        "BA": (-12.9718, -38.5011), "CE": (-3.71722, -38.5434),
    "DF": (-15.7934, -47.8823),   "ES": (-20.3155, -40.3128), "GO": (-16.6869, -49.2648),
    "MA": (-2.53, -44.30),        "MG": (-19.9167, -43.9345), "MS": (-20.4697, -54.6201),
    "MT": (-15.6010, -56.0974),   "PA": (-1.4558, -48.5039),  "PB": (-7.1216, -34.8829),
    "PE": (-8.0476, -34.8770),    "PI": (-5.09, -42.80),      "PR": (-25.4284, -49.2733),
    "RJ": (-22.9068, -43.1729),   "RN": (-5.7945, -35.2110),  "RO": (-8.76, -63.90),
    "RR": (2.82, -60.67),         "RS": (-30.0346, -51.2177), "SC": (-27.5949, -48.5482),
    "SE": (-10.9472, -37.0731),   "SP": (-23.5505, -46.6333), "TO": (-10.1846, -48.3336),
}

# Links de referência por UF (quando não houver um site oficial mapeado, cai para uma busca)
UF_LINKS = {
    "AC": "https://www.google.com/search?q=hemocentro+AC",
    "AL": "https://www.google.com/search?q=hemocentro+AL",
    "AM": "https://www.google.com/search?q=hemocentro+AM",
    "AP": "https://www.google.com/search?q=hemocentro+AP",
    "BA": "https://www.google.com/search?q=hemocentro+BA",
    "CE": "https://www.google.com/search?q=hemocentro+CE",
    "DF": "https://www.google.com/search?q=hemocentro+DF",
    "ES": "https://www.google.com/search?q=hemocentro+ES",
    "GO": "https://www.google.com/search?q=hemocentro+GO",
    "MA": "https://www.google.com/search?q=hemocentro+MA",
    "MG": "https://www.google.com/search?q=hemocentro+MG",
    "MS": "https://www.google.com/search?q=hemocentro+MS",
    "MT": "https://www.google.com/search?q=hemocentro+MT",
    "PA": "https://www.google.com/search?q=hemocentro+PA",
    "PB": "https://www.google.com/search?q=hemocentro+PB",
    "PE": "https://www.google.com/search?q=hemocentro+PE",
    "PI": "https://www.google.com/search?q=hemocentro+PI",
    "PR": "https://www.google.com/search?q=hemocentro+PR",
    "RJ": "https://www.google.com/search?q=hemocentro+RJ",
    "RN": "https://www.google.com/search?q=hemocentro+RN",
    "RO": "https://www.google.com/search?q=hemocentro+RO",
    "RR": "https://www.google.com/search?q=hemocentro+RR",
    "RS": "https://www.google.com/search?q=hemocentro+RS",
    "SC": "https://www.google.com/search?q=hemocentro+SC",
    "SE": "https://www.google.com/search?q=hemocentro+SE",
    "SP": "https://www.google.com/search?q=hemocentro+SP",
    "TO": "https://www.google.com/search?q=hemocentro+TO",
}

# URL padrão do Hemoprod (ANVISA)
URL_HEMOPROD_DEFAULT = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

# ------------------------
# Sidebar (logo + navegação)
# ------------------------
with st.sidebar:
    logo_path = "ativos/logo.svg"
    if os.path.exists(logo_path):
        st.image(logo_path, use_column_width=True)
    else:
        st.markdown("### 🩸 VisioData")

    st.markdown("#### Navegação")
    secao = st.radio(
        "Escolha a seção",
        ["ANVISA (nacional)", "Estoques estaduais", "Cadastrar doador", "Sobre"],
        index=0,
        key="nav_radio"
    )

# ------------------------
# 1) ANVISA (nacional)
# ------------------------
if secao == "ANVISA (nacional)":
    st.title("Painel de Estoques e Produção Hemoterápica — ANVISA (Hemoprod)")

    st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

    url = st.text_input(
        "URL do CSV (Hemoprod — ANVISA)",
        value=URL_HEMOPROD_DEFAULT,
        key="carregar_url"
    )

    cols_a = st.columns([1, 6, 1])
    with cols_a[1]:
        bt = st.button("Carregar agora", type="primary", use_container_width=True, key="carregar_btn")

    df = None
    if bt and url.strip():
        with st.spinner("Baixando e processando o CSV..."):
            df = read_csv_robusto(url.strip())
            st.success(f"Base carregada: **{len(df):,}** linhas × **{df.shape[1]}** colunas.", icon="✅")

    # Alternativa: upload manual
    with st.expander("...ou envie o CSV (alternativa)"):
        up = st.file_uploader("Upload CSV (máx. 200MB)", type=["csv"], key="upload_csv")
        if up is not None:
            with st.spinner("Lendo o arquivo enviado..."):
                df = read_csv_robusto(up)
                st.success(f"Base carregada: **{len(df):,}** linhas × **{df.shape[1]}** colunas.", icon="✅")

    if df is None:
        st.info("Use **Carregar agora** (ou faça upload) para visualizar KPIs e mapa.")
        st.stop()

    # Amostra
    with st.expander("Amostra (100 linhas)"):
        st.dataframe(df.head(100), use_container_width=True)

    # ---------------- KPIs automáticos ----------------
    st.markdown("### KPIs automáticos")

    cols_lista = list(df.columns)
    num_cols = [c for c in cols_lista if pd.api.types.is_numeric_dtype(df[c])]

    ano_detected = detectar_coluna_ano(df)
    uf_detected = detectar_coluna_uf(df)
    metrica_detected = detectar_coluna_metrica(df)

    kcol1, kcol2, kcol3 = st.columns(3)

    with kcol1:
        ano_kpi = st.selectbox(
            "Coluna de ano (se não detectar)",
            options=make_options(ano_detected, cols_lista),
            index=0,
            key="kpi_ano"
        )
    with kcol2:
        uf_kpi = st.selectbox(
            "Coluna UF (se existir)",
            options=make_options(uf_detected, cols_lista),
            key="kpi_uf"
        )
    with kcol3:
        metrica_kpi = st.selectbox(
            "Coluna MÉTRICA (numérica)",
            options=make_options(metrica_detected, num_cols) if num_cols else ["(nenhuma numérica encontrada)"],
            key="kpi_metrica"
        )

    ufs_distintas = df[uf_kpi].nunique() if uf_kpi in df.columns else 0
    st.caption(f"**Registros:** {len(df):,} | **UF distintas:** {ufs_distintas}")

    # Tabela agregada para KPIs
    if uf_kpi in df.columns and metrica_kpi in df.columns:
        # tenta converter métrica
        df_tmp = df.copy()
        df_tmp[metrica_kpi] = pd.to_numeric(df_tmp[metrica_kpi], errors="coerce")
        grp = (
            df_tmp.groupby([uf_kpi], dropna=False)[metrica_kpi]
            .sum(min_count=1)
            .reset_index()
            .rename(columns={uf_kpi: "uf", metrica_kpi: "valor"})
            .sort_values("uf")
        )
        st.dataframe(grp, use_container_width=True, height=320)

    st.markdown("---")

    # ---------------- Mapa por UF ----------------
    st.markdown("### Mapa por UF")

    mcol1, mcol2, mcol3 = st.columns(3)
    with mcol1:
        ano_map = st.selectbox(
            "Coluna de ano (se não detectar)",
            options=make_options(ano_detected, cols_lista),
            index=0,
            key="map_ano"
        )
    with mcol2:
        uf_map = st.selectbox(
            "Coluna UF",
            options=make_options(uf_detected, cols_lista),
            key="map_uf"
        )
    with mcol3:
        valor_map = st.selectbox(
            "Coluna de valor (coletas/uso, etc)",
            options=make_options(metrica_detected, num_cols) if num_cols else ["(nenhuma numérica encontrada)"],
            key="map_valor"
        )

    if uf_map not in df.columns or valor_map not in df.columns:
        st.warning("Sem dados suficientes para o mapa (verifique UF e métrica).")
        st.stop()

    df_m = df[[uf_map, valor_map]].copy()
    df_m[valor_map] = pd.to_numeric(df_m[valor_map], errors="coerce")
    df_m = (
        df_m.groupby(uf_map, dropna=False)[valor_map]
        .sum(min_count=1)
        .reset_index()
        .rename(columns={uf_map: "uf", valor_map: "valor"})
    )

    # normaliza UF para 2 letras
    df_m["uf"] = df_m["uf"].astype(str).str.strip().str.upper().str[:2]
    df_m = df_m[df_m["uf"].isin(UF_LATLON.keys())]

    if df_m.empty:
        st.warning("Não foi possível montar o mapa: não encontrei UFs válidas (siglas).")
        st.stop()

    # acrescenta lat/lon
    df_m["lat"] = df_m["uf"].map(lambda x: UF_LATLON[x][0])
    df_m["lon"] = df_m["uf"].map(lambda x: UF_LATLON[x][1])
    df_m["link"] = df_m["uf"].map(lambda x: UF_LINKS.get(x, "https://www.google.com/"))

    # escala de tamanho
    vmin, vmax = df_m["valor"].min(skipna=True), df_m["valor"].max(skipna=True)
    if pd.isna(vmin) or pd.isna(vmax) or vmax <= 0:
        df_m["size"] = 1000
    else:
        # normaliza 1k a 25k
        df_m["size"] = ( (df_m["valor"] - vmin) / (vmax - vmin + 1e-9) * (25000 - 1500) + 1500 ).fillna(1500)

    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(latitude=-14.2350, longitude=-51.9253, zoom=3.7, pitch=0),
            map_style="mapbox://styles/mapbox/dark-v11",
            layers=[
                pdk.Layer(
                    "ScatterplotLayer",
                    data=df_m,
                    get_position="[lon, lat]",
                    get_radius="size",
                    get_fill_color=[255, 60, 60, 160],
                    pickable=True,
                    auto_highlight=True,
                )
            ],
            tooltip={"text": "{uf}: {valor}"},
        ),
        use_container_width=True,
        height=540
    )

    # tabela com link clicável
    df_links = df_m[["uf", "valor", "link"]].sort_values("uf").reset_index(drop=True)
    df_links["link"] = df_links["link"].apply(lambda u: f"[Abrir]({u})")
    st.markdown("#### Tabela por UF (com link)")
    st.dataframe(df_links, use_container_width=True, height=360)

# ------------------------
# 2) Estoques estaduais (links)
# ------------------------
elif secao == "Estoques estaduais":
    st.title("Estoques estaduais — referências úteis")

    st.caption("Abaixo, links de referência por UF. Caso não haja um site oficial mapeado, o link abre uma busca segura do Google para 'hemocentro + UF'.")

    ufs = sorted(UF_LINKS.keys())
    ncol = 6
    rows = (len(ufs) + ncol - 1) // ncol
    for r in range(rows):
        cols = st.columns(ncol)
        for c in range(ncol):
            i = r * ncol + c
            if i >= len(ufs): break
            uf = ufs[i]
            with cols[c]:
                st.markdown(f"**{uf}**  \n[Ir para o site/consulta]({UF_LINKS[uf]})", unsafe_allow_html=True)

# ------------------------
# 3) Cadastrar doador
# ------------------------
elif secao == "Cadastrar doador":
    st.title("Cadastro de possíveis doadores")
    st.caption("Se você deseja ser notificado por campanhas locais, preencha seus dados. (Uso acadêmico/demonstração.)")

    with st.form("form_doador", clear_on_submit=True):
        col1, col2 = st.columns(2)
        with col1:
            nome = st.text_input("Nome completo *", key="doa_nome")
            email = st.text_input("E-mail *", key="doa_email")
            telefone = st.text_input("Telefone", key="doa_tel")
        with col2:
            uf = st.selectbox("UF *", sorted(UF_LATLON.keys()), key="doa_uf")
            cidade = st.text_input("Cidade", key="doa_cidade")
            tipo = st.selectbox("Tipo sanguíneo", ["Não informado","O-","O+","A-","A+","B-","B+","AB-","AB+"], key="doa_tipo")
        aceite = st.checkbox("Autorizo o contato para campanhas e declaro ciência da finalidade (LGPD).", key="doa_lgpd")
        enviado = st.form_submit_button("Cadastrar", type="primary")

    if enviado:
        if not nome or not email or not uf or not aceite:
            st.error("Preencha os campos marcados com * e aceite a declaração.")
        else:
            registro = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                "nome": nome, "email": email, "telefone": telefone,
                "uf": uf, "cidade": cidade, "tipo": tipo
            }
            # guarda numa "base" local (em memória + opção de download)
            if "doadores" not in st.session_state:
                st.session_state.doadores = []
            st.session_state.doadores.append(registro)
            st.success("Cadastro enviado com sucesso! Obrigado. 🙏")

    st.markdown("#### Registros desta sessão")
    if "doadores" in st.session_state and st.session_state.doadores:
        df_d = pd.DataFrame(st.session_state.doadores)
        st.dataframe(df_d, use_container_width=True, height=320)

        csv = df_d.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Baixar inscrições (CSV)",
            data=csv,
            file_name="doadores_visiodata.csv",
            mime="text/csv"
        )
    else:
        st.info("Sem cadastros ainda.")

# ------------------------
# 4) Sobre
# ------------------------
else:
    st.title("Sobre o VisioData")
    st.markdown(
        """
**VisioData** é um painel didático para explorar dados públicos do sistema **Hemoprod (ANVISA)**.

**Recursos:**
- Botão **Carregar agora** com leitura robusta do CSV oficial (detecção de separador, tolerância a linhas com falhas, sem `low_memory`).
- **KPIs automáticos** com seleção assistida de colunas (ano, UF, métrica).
- **Mapa por UF** (pydeck) usando centróides e escala de bolhas pelo valor selecionado.
- **Links por UF** na seção *Estoques estaduais*.
- **Cadastro de doadores** (demonstração) com exportação para CSV.

Este projeto é **não-oficial** e tem finalidade **educacional**/apresentações.
"""
    )
