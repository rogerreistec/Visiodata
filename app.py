# VisioData — Estoques e Produção Hemoterápica (Brasil)
# -----------------------------------------------------
# Rodar local: streamlit run app.py
# Requisitos: ver requirements.txt

import io
import time
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
from pathlib import Path

# ------------- CONFIG / ESTILO ----------------------------------------------

st.set_page_config(page_title="VisioData – Estoques e Produção Hemoterápica", layout="wide")

CUSTOM_CSS = """
<style>
.title-row { display:flex; align-items:center; gap:.75rem; }
.pill { background:#E10600; color:white; padding:.25rem .6rem; border-radius:999px; font-weight:700; }
.muted { color:#6B7280; }
footer { visibility:hidden; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

c1, c2 = st.columns([1,6])
with c1:
    st.write(":drop_of_blood:")
with c2:
    st.markdown(
        '<div class="title-row"><span class="pill">VisioData</span>'
        "<h2 style='margin:0'>Painel de Estoques e Produção Hemoterápica</h2></div>",
        unsafe_allow_html=True,
    )
st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

# ------------- UTILS --------------------------------------------------------

@st.cache_data(show_spinner=False)
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out

@st.cache_data(show_spinner=True)
def load_csv_robust(url_or_file, sep=None, encodings=("utf-8", "latin-1")) -> pd.DataFrame:
    """
    Lê CSV local/upload/URL tentando separadores e encodings comuns.
    """
    errs = []
    for enc in encodings:
        for sep_try in (sep, ",", ";", "|"):
            if sep_try is None:
                continue
            try:
                df = pd.read_csv(url_or_file, sep=sep_try, encoding=enc, engine="python")
                return df
            except Exception as e:
                errs.append(f"[enc={enc} sep={sep_try}]: {e}")
    # última tentativa sem forçar sep
    for enc in encodings:
        try:
            df = pd.read_csv(url_or_file, encoding=enc)
            return df
        except Exception as e:
            errs.append(f"[enc={enc} sep=auto]: {e}")
    raise ValueError("Falha ao ler CSV -> " + " | ".join(errs))

def first_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

# Mapa: coordenadas aproximadas por UF (centroides do estado)
UF_COORD = {
    'AC': (-9.02,  -70.81), 'AL': (-9.62,  -36.82), 'AM': (-4.13,  -60.02),
    'AP': (0.04,   -51.05), 'BA': (-12.97, -38.51), 'CE': (-5.18,  -39.45),
    'DF': (-15.78, -47.93), 'ES': (-20.32, -40.34), 'GO': (-16.64, -49.25),
    'MA': (-4.23,  -43.94), 'MG': (-18.10, -44.62), 'MS': (-20.48, -54.62),
    'MT': (-12.64, -55.42), 'PA': (-1.41,  -48.44), 'PB': (-7.12,  -36.72),
    'PE': (-8.05,  -34.90), 'PI': (-8.03,  -42.42), 'PR': (-25.43, -49.27),
    'RJ': (-22.91, -43.17), 'RN': (-5.79,  -35.21), 'RO': (-11.22, -62.80),
    'RR': (2.82,   -60.67), 'RS': (-30.03, -51.23), 'SC': (-27.59, -48.55),
    'SE': (-10.91, -37.07), 'SP': (-23.55, -46.64), 'TO': (-10.18, -48.33),
}

# População (aprox.) para taxa/100k (IBGE arredondado)
POP_UF = {
    "AC": 0.906e6, "AL": 3.377e6, "AM": 4.269e6, "AP": 0.877e6, "BA": 14.87e6,
    "CE": 9.26e6, "DF": 3.10e6, "ES": 4.10e6, "GO": 7.29e6, "MA": 6.77e6,
    "MG": 20.6e6, "MS": 2.75e6, "MT": 3.78e6, "PA": 8.69e6, "PB": 4.03e6,
    "PE": 9.60e6, "PI": 3.29e6, "PR": 11.44e6, "RJ": 16.05e6, "RN": 3.50e6,
    "RO": 1.86e6, "RR": 0.73e6, "RS": 10.88e6, "SC": 7.61e6, "SE": 2.36e6,
    "SP": 44.42e6, "TO": 1.61e6
}

# Links úteis por UF (oficiais onde conhecido; genéricos para as demais)
LINKS_UF = {
    "SP": {"Pró-Sangue (SP)": "https://www.prosangue.sp.gov.br/"},
    "RJ": {"HEMORIO (RJ)": "https://www.hemorio.rj.gov.br/"},
}
# para as demais UFs, criaremos links genéricos: site SES (busca), portal gov.br do estado
for uf in UF_COORD:
    if uf not in LINKS_UF:
        LINKS_UF[uf] = {
            f"Secretaria de Saúde {uf} (busca)": f"https://www.google.com/search?q=hemocentro+{uf}",
            f"Portal gov.br {uf} (busca)": f"https://www.google.com/search?q=doa%C3%A7%C3%A3o+de+sangue+{uf}"
        }

# ------------- SIDEBAR ------------------------------------------------------

st.sidebar.header("Navegação")
page = st.sidebar.radio(
    "Escolha a seção",
    ["ANVISA (nacional)", "Estoques estaduais", "Cadastro de doadores", "Sobre"],
    index=0
)

# ------------- PÁGINA: ANVISA ----------------------------------------------

DEFAULT_URL = (
    "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
    "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
    "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
)

if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")

    url = st.text_input("URL do CSV (Hemoprod — ANVISA)", value=DEFAULT_URL, label_visibility="visible")
    c_btn1, c_btn2 = st.columns([1,2])
    with c_btn1:
        go = st.button("Carregar agora", type="primary")
    with c_btn2:
        st.caption("...ou envie o CSV (alternativa)")

    upfile = st.file_uploader("Drag and drop file here", type=["csv"], label_visibility="collapsed")

    df_raw = None
    try:
        if go and url.strip():
            df_raw = load_csv_robust(url, sep=",")
        elif upfile is not None:
            df_raw = load_csv_robust(upfile, sep=",")
    except Exception as e:
        st.error(f"Erro ao carregar: {e}")

    if df_raw is not None:
        st.success(f"Base carregada: {len(df_raw):,} linhas × {df_raw.shape[1]} colunas.")
        df = normalize_columns(df_raw)

        with st.expander("Amostra (100 linhas)", expanded=False):
            st.dataframe(df.head(100), use_container_width=True, height=360)

        tabs = st.tabs(["KPIs automáticos", "Mapa por UF"])
        with tabs[0]:
            # Autodetectar possíveis colunas
            ano_col = first_col(df, ["ano de referência", "ano", "ano_referencia", "ano referencia"])
            uf_col = first_col(df, ["uf", "estado", "sigla_uf", "unidade_federativa"])
            # Métricas possíveis por nome
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            # Sugestões comumente encontradas
            metric_coleta = first_col(df, ["coletas", "qtd_coletas", "coleta_total"])
            metric_uso = first_col(df, ["transfusões", "transfusoes", "qtd_transfusões", "qtd_transfusoes", "uso_total"])

            sel_ano = st.selectbox("Coluna de ano (se não detectar)", [ano_col] + [None] + list(df.columns), index=0 if ano_col else 1)
            sel_uf  = st.selectbox("Coluna UF (se existir)", [uf_col] + [None] + list(df.columns), index=0 if uf_col else 1)

            # Onde não detectado, oferecemos uma lista curta de métricas
            default_metric = metric_coleta or metric_uso or (num_cols[0] if num_cols else None)
            options_metric = ([default_metric] + num_cols) if default_metric else num_cols
            sel_metric = st.selectbox("Coluna MÉTRICA (numérica)", options_metric)

            group = []
            if sel_ano: group.append(sel_ano)
            if sel_uf and sel_uf in df.columns: group.append(sel_uf)

            if sel_metric:
                g = df.groupby(group, dropna=False)[sel_metric].sum(numeric_only=True).reset_index().rename(columns={sel_metric:"valor"})
                c1, c2 = st.columns([2,2])
                with c1:
                    st.metric("Registros", f"{len(df):,}")
                    st.write("Colunas:", ", ".join(df.columns[:30]))
                with c2:
                    if sel_uf and sel_uf in df.columns:
                        st.metric("UF distintas", f"{df[sel_uf].nunique():,}")
                st.dataframe(g, use_container_width=True, height=420)
                # taxa por 100k se houver UF
                if sel_uf and sel_uf in g.columns:
                    g2 = g.copy()
                    g2["uf"] = g2[sel_uf].astype(str).str.upper().str[:2]
                    if "uf" in g2.columns:
                        g2["pop"] = g2["uf"].map(POP_UF).fillna(np.nan)
                        g2["taxa_100k"] = (g2["valor"] / g2["pop"] * 100000).round(4)
                        st.caption("Taxa por 100k (se houver população mapeada para a UF):")
                        st.dataframe(g2, use_container_width=True, height=320)
            else:
                st.info("Selecione uma coluna numérica para calcular os KPIs.")

        with tabs[1]:
            st.caption("Selecione uma UF e uma coluna métrica (numérica) para renderizar o mapa.")
            ano_col = first_col(df, ["ano de referência","ano","ano_referencia","ano referencia"])
            uf_col  = first_col(df, ["uf","estado","sigla_uf","unidade_federativa"])
            num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            sel_ano = st.selectbox("Coluna de ano (se não detectar)", [ano_col] + [None] + list(df.columns), index=0 if ano_col else 1, key="map_ano")
            sel_uf  = st.selectbox("Coluna UF", [uf_col] + [None] + list(df.columns), index=0 if uf_col else 1, key="map_uf")
            sel_val = st.selectbox("Coluna de valor (coletas/uso, etc)", num_cols, key="map_val")

            if sel_uf and sel_val and sel_uf in df.columns:
                # agregamos por UF (e por ano, se informado)
                group = [sel_uf]
                if sel_ano: group.append(sel_ano)
                g = df.groupby(group, dropna=False)[sel_val].sum(numeric_only=True).reset_index().rename(columns={sel_val:"valor"})

                # Usar última informação por UF se houver ano
                if sel_ano and sel_ano in g.columns:
                    g = g.sort_values(sel_ano).groupby(sel_uf).tail(1)

                # prepara pontos
                pts = []
                for _, r in g.iterrows():
                    uf = str(r[sel_uf]).upper()[:2]
                    if uf in UF_COORD:
                        lat, lon = UF_COORD[uf]
                        pts.append({"position":[lon, lat], "uf":uf, "valor":float(r["valor"])})
                if pts:
                    layer = pdk.Layer(
                        "ScatterplotLayer",
                        data=pts,
                        get_position="position",
                        get_radius="valor",
                        radius_scale=0.5, radius_min_pixels=5,
                        pickable=True,
                        get_fill_color=[225,0,0,140],
                    )
                    view = pdk.ViewState(latitude=-14.235, longitude=-51.9253, zoom=3)
                    st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text":"{uf}: {valor}"}))
                else:
                    st.info("Sem dados suficientes para o mapa (verifique UF e métrica).")
            else:
                st.info("Para o mapa, preciso de UF e uma métrica numérica.")

# ------------- PÁGINA: ESTOQUES ESTADUAIS ----------------------------------

elif page == "Estoques estaduais":
    st.subheader("Produção hemoterápica – ANVISA (Hemoprod)")

    st.write("Envie um CSV **estadual** com colunas: `data`, `uf`, `hemocentro`, `tipo`, `rh`, `estoque_atual`, `estoque_minimo`.")
    up_state = st.file_uploader("Enviar CSV estadual", type=["csv"])
    df_state = None
    if up_state is not None:
        try:
            df_state = load_csv_robust(up_state, sep=",")
        except Exception as e:
            st.error(f"Erro ao ler: {e}")

    # Links por UF
    st.markdown("### Links por UF (doação, hemocentros, informações oficiais)")
    grid = st.columns(4)
    ufs = sorted(UF_COORD.keys())
    for i, uf in enumerate(ufs):
        with grid[i % 4]:
            st.markdown(f"**{uf}**")
            for label, url in LINKS_UF.get(uf, {}).items():
                st.link_button(label, url, use_container_width=True, type="secondary")

    if df_state is not None:
        st.markdown("### Amostra dos dados enviados")
        st.dataframe(normalize_columns(df_state).head(200), use_container_width=True, height=360)

        # Indicadores simples se existir estoque
        df2 = normalize_columns(df_state)
        need_cols = ["uf","estoque_atual","estoque_minimo"]
        if all(c in df2.columns for c in need_cols):
            g = df2.groupby("uf")[["estoque_atual","estoque_minimo"]].sum(numeric_only=True)
            g["cobertura"] = (g["estoque_atual"] / g["estoque_minimo"]).replace([np.inf,-np.inf], np.nan)
            st.markdown("### Cobertura por UF (estoque atual / estoque mínimo)")
            st.dataframe(g.reset_index(), use_container_width=True, height=360)

            # alerta simples
            criticos = g[g["cobertura"] < 0.8].index.tolist()
            if criticos:
                st.error("Cobertura crítica (< 0,8) em: " + ", ".join(criticos))
            elif not g.empty:
                st.success("Nenhuma UF em nível crítico segundo o arquivo enviado.")

# ------------- PÁGINA: CADASTRO DE DOADORES --------------------------------

elif page == "Cadastro de doadores":
    st.subheader("Cadastro de possíveis doadores")

    # estado em memória (sessão) — no Streamlit Cloud é efêmero
    if "donors" not in st.session_state:
        st.session_state["donors"] = []

    with st.form("donor_form", clear_on_submit=True):
        c1, c2 = st.columns(2)
        with c1:
            nome = st.text_input("Nome completo *")
            email = st.text_input("E-mail *")
            telefone = st.text_input("Telefone/WhatsApp")
        with c2:
            uf = st.selectbox("UF *", sorted(UF_COORD.keys()))
            cidade = st.text_input("Cidade")
            tipo_sanguineo = st.selectbox("Tipo sanguíneo *", ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"])
        consent = st.checkbox("Aceito ser contatado/a por hemocentros/serviços oficiais.", value=True)
        submitted = st.form_submit_button("Cadastrar doador", type="primary")

    if submitted:
        if nome and email and uf and tipo_sanguineo and consent:
            st.session_state["donors"].append(
                {
                    "data": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "nome": nome, "email": email, "telefone": telefone,
                    "cidade": cidade, "uf": uf, "tipo_sanguineo": tipo_sanguineo,
                    "consentimento": bool(consent)
                }
            )
            st.success("Cadastro recebido! Obrigado por se voluntariar ❤️")
        else:
            st.warning("Preencha os campos obrigatórios (*) e aceite o consentimento.")

    if st.session_state["donors"]:
        st.markdown("### Banco local de doadores (sessão)")
        dfd = pd.DataFrame(st.session_state["donors"])
        st.dataframe(dfd, use_container_width=True, height=360)

        # botão de download
        csv = dfd.to_csv(index=False).encode("utf-8")
        st.download_button("Baixar CSV dos cadastros", data=csv, file_name="cadastro_doadores.csv", mime="text/csv")

        # contagens rápidas por UF / tipo
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Contagem por UF")
            st.dataframe(dfd["uf"].value_counts().rename_axis("uf").reset_index(name="cadastros"), use_container_width=True, height=300)
        with c2:
            st.caption("Contagem por tipo sanguíneo")
            st.dataframe(dfd["tipo_sanguineo"].value_counts().rename_axis("tipo").reset_index(name="cadastros"), use_container_width=True, height=300)
    else:
        st.info("Nenhum cadastro nesta sessão ainda.")

# ------------- PÁGINA: SOBRE ------------------------------------------------

elif page == "Sobre":
    st.subheader("Sobre o projeto")
    st.write(
        """
        **VisioData** é um painel em Streamlit para exploração de dados públicos sobre
        produção e estoques hemoterápicos no Brasil.  
        - **ANVISA (Hemoprod):** leitura direta via URL oficial ou upload do CSV  
        - **KPIs e mapa por UF:** agregados simples e visualização geográfica  
        - **Estoques estaduais:** upload do arquivo do seu estado para cálculo de cobertura  
        - **Cadastro de doadores:** formulário voluntário (armazenamento temporário na sessão)  

        O propósito é **acadêmico** e para **divulgação**: sempre cite a **fonte ANVISA**
        ou os órgãos estaduais quando compartilhar os resultados.
        """
    )
    st.caption("Autor: você. Repositório: GitHub → rogerreistec/Visiodata")
