# VisioData — Estoques e Produção Hemoterápica (Brasil)
# ------------------------------------------------------
# Requisitos (pip): streamlit, pandas, numpy, pydeck
# Rodar local (opcional): streamlit run app.py

import io
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk

st.set_page_config(
    page_title="VisioData — Estoques e Produção Hemoterápica",
    layout="wide",
)

# =============== ESTILO / CABEÇALHO ===============
CUSTOM_CSS = """
<style>
.title-row { display:flex; align-items:center; gap:.75rem; }
.pill { background:#E10600; color:white; padding:.25rem .6rem; border-radius:999px; font-weight:700; }
.muted { color:#6B7280; }
footer { visibility:hidden; } /* esconde rodapé streamlit */
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

c1, c2 = st.columns([1, 6])
with c1:
    # tenta achar um logo em várias pastas/nomes
    LOGOS = [
        "ativos/logo.svg","ativos/logo.png","assets/logo.svg","assets/logo.png",
        "logo.svg","logo.png"
    ]
    showed_logo = False
    for p in LOGOS:
        if Path(p).exists():
            st.image(p, width=120)
            showed_logo = True
            break
    if not showed_logo:
        st.write(":drop_of_blood:")

with c2:
    st.markdown(
        '<div class="title-row"><span class="pill">VisioData</span>'
        '<h2 style="margin:0">Painel de Estoques e Produção Hemoterápica</h2></div>',
        unsafe_allow_html=True,
    )
    st.caption("Fontes oficiais e dados agregados — pronto para apresentação acadêmica.")

# =============== HELPERS ===============
UF_COORD = {
    "AC": (-70.55, -9.02), "AL": (-36.28, -9.62), "AM": (-62.00, -3.47), "AP": (-51.07, 0.04),
    "BA": (-41.55, -12.97), "CE": (-39.26, -5.79), "DF": (-47.86, -15.79), "ES": (-40.34, -19.58),
    "GO": (-49.25, -16.64), "MA": (-45.78, -5.42), "MG": (-44.56, -18.10), "MS": (-54.62, -20.48),
    "MT": (-56.10, -12.64), "PA": (-52.47, -4.47), "PB": (-36.78, -7.12), "PE": (-36.90, -8.05),
    "PI": (-42.80, -8.64), "PR": (-51.23, -24.89), "RJ": (-42.52, -22.84), "RN": (-36.51, -5.79),
    "RO": (-62.80, -10.83), "RR": (-61.40, 2.82), "RS": (-53.21, -30.03), "SC": (-48.55, -27.59),
    "SE": (-37.07, -10.91), "SP": (-46.64, -23.55), "TO": (-48.33, -10.18),
}

NOME2UF = {
    'acre':'AC','alagoas':'AL','amazonas':'AM','amapá':'AP','bahia':'BA','ceará':'CE','distrito federal':'DF',
    'espírito santo':'ES','goiás':'GO','maranhão':'MA','minas gerais':'MG','mato grosso do sul':'MS',
    'mato grosso':'MT','pará':'PA','paraíba':'PB','pernambuco':'PE','piauí':'PI','paraná':'PR',
    'rio de janeiro':'RJ','rio grande do norte':'RN','rondônia':'RO','roraima':'RR','rio grande do sul':'RS',
    'santa catarina':'SC','sergipe':'SE','são paulo':'SP','tocantins':'TO'
}

# População aprox. (milhares) — só para taxa por 100k hab.
POP_UF = {
    "AC": 906, "AL": 3354, "AM": 4209, "AP": 877, "BA": 14876, "CE": 9012, "DF": 3106, "ES": 4106,
    "GO": 7296, "MA": 6776, "MG": 20566, "MS": 2756, "MT": 3786, "PA": 8698, "PB": 4036, "PE": 9206,
    "PI": 3282, "PR": 11446, "RJ": 16056, "RN": 3506, "RO": 1856, "RR": 736, "RS": 10886, "SC": 7616,
    "SE": 2366, "SP": 44266, "TO": 1616,
}

def _first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols = [c for c in df.columns]
    low = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in low: 
            return low[cand]
    for c in cols:
        if any(x in c.lower() for x in candidates):
            return c
    return None

def _coerce_uf(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    su = s.str.upper()
    # já está UF de 2 chars?
    ok = su.where(su.str.len()==2)
    miss = ok.isna()
    if miss.any():
        lowered = s.str.lower()
        mapped = lowered.map(NOME2UF)
        ok = ok.fillna(mapped.str.upper())
    return ok

@st.cache_data(show_spinner=False, ttl=3600)
def _load_csv_robust(url_or_buf, encodings=("utf-8","latin-1"), seps=(",", ";", "\t")) -> pd.DataFrame:
    errs = []
    for enc in encodings:
        for sep in seps:
            try:
                df = pd.read_csv(url_or_buf, encoding=enc, sep=sep, engine="python")
                # normaliza colunas
                df.columns = [c.strip() for c in df.columns]
                return df
            except Exception as e:
                errs.append(f"{enc}/{sep}: {e}")
                continue
    raise ValueError("Falha ao ler CSV — " + " | ".join(errs))

def _numericize(series: pd.Series) -> pd.Series:
    """Converte strings para número robustamente (corrige SP/RJ=0 quando há lixo)"""
    s = series.astype(str).str.replace("\u00a0","", regex=False).str.strip()
    # remove milhares '.' e troca vírgula por ponto se existir
    s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    out = pd.to_numeric(s, errors="coerce")
    return out

# =============== NAVEGAÇÃO ===============
st.sidebar.header("Navegação")
page = st.sidebar.radio("Escolha a seção", ["ANVISA (nacional)", "Estoques estaduais", "Sobre"], index=0)

# =============== PÁGINA ANVISA (principal) ===============
if page == "ANVISA (nacional)":
    st.subheader("Produção hemoterápica — ANVISA (Hemoprod)")
    st.caption("Dados brutos nacionais (CSV) a partir de 2022 — ver página oficial para dicionário e atualizações.")

    DEFAULT_URL = (
        "https://www.gov.br/anvisa/pt-br/centraisdeconteudo/publicacoes/"
        "sangue-tecidos-celulas-e-orgaos/producao-e-avaliacao-de-servicos-de-hemoterapia/"
        "dados-brutos-de-producao-hemoterapica-1/hemoprod_nacional.csv"
    )

    url = st.text_input("URL do CSV nacional (Hemoprod — ANVISA)", value=DEFAULT_URL)
    up = st.file_uploader("...ou envie um CSV (alternativa)", type=["csv"])

    # carrega base
    try:
        if up is not None:
            df = _load_csv_robust(up)  # upload
        else:
            df = _load_csv_robust(url)  # remoto

        # normaliza e preenche colunas úteis
        # detecta potenciais colunas
        cand_ano = ["ano", "ano de referência", "ano_referencia", "ano referencia", "ano referência"]
        cand_uf  = ["uf", "unidade federativa", "estado", "sigla uf"]

        ano_col = _first_col(df, cand_ano) or st.selectbox("Escolha a coluna de ano", list(df.columns))
        if ano_col not in df.columns:
            st.warning("Não encontrei uma coluna de ano.")
        uf_col_guess = _first_col(df, cand_uf)

        # coluna UF se existir -> coerção para sigla
        if uf_col_guess:
            df["uf"] = _coerce_uf(df[uf_col_guess])
        elif "uf" not in df.columns:
            df["uf"] = np.nan  # deixa em branco para não quebrar

        # tenta encontrar colunas com quantidade/valor/coletas/uso
        num_candidates = []
        for c in df.columns:
            if df[c].dtype.kind in "iufc":  # já numérico
                num_candidates.append(c)
            else:
                # se é texto mas parece número, tentaremos converter mais tarde
                if any(k in c.lower() for k in ["valor","quant","colet","uso","transfus"]):
                    num_candidates.append(c)

        # escolha de colunas (e conversão numérica robusta)
        st.markdown("### KPIs automáticos")
        c_top = st.columns(3)
        with c_top[0]:
            sel_ano_col = st.selectbox("Coluna de ano (se não detectar)", options=[ano_col] + [c for c in df.columns if c!=ano_col], index=0)
        with c_top[1]:
            sel_uf_col = st.selectbox("Coluna UF (se existir)", options=["uf"] + [c for c in df.columns if c != "uf"], index=0)
        with c_top[2]:
            # métrica padrão: tenta valor/coletas/uso; caso contrário, qualquer numérica
            default_metric = None
            for key in ["colet", "uso", "transfus", "valor", "quant"]:
                pick = [c for c in num_candidates if key in c.lower()]
                if pick:
                    default_metric = pick[0]; break
            sel_metric = st.selectbox(
                "Coluna MÉTRICA (numérica)",
                options=num_candidates if num_candidates else list(df.columns),
                index=(num_candidates.index(default_metric) if default_metric in (num_candidates or []) else 0)
            )

        # normalizações que afetam SP/RJ = 0 se o dado veio como string
        df[sel_metric] = _numericize(df[sel_metric]).fillna(0)
        # ano como inteiro (quando possível)
        try:
            df[sel_ano_col] = pd.to_numeric(df[sel_ano_col], errors="coerce").astype("Int64")
        except Exception:
            pass

        with st.expander("Amostra (100 linhas)"):
            st.dataframe(df.head(100), use_container_width=True)

        # filtros
        col_filters = st.columns(3)
        # lista de anos válidos
        anos_validos = [a for a in sorted(df[sel_ano_col].dropna().unique()) if str(a) != "nan"]
        with col_filters[0]:
            ano_sel = st.selectbox("Filtrar por ano", options=["Todos"] + list(anos_validos), index=0)
        with col_filters[1]:
            ufs_validas = [u for u in sorted(df[sel_uf_col].dropna().unique()) if isinstance(u,str)]
            uf_sel = st.multiselect("Filtrar por UF", options=ufs_validas, default=[])
        with col_filters[2]:
            agg_op = st.selectbox("Como agrego a métrica?", ["Soma", "Média", "Máximo", "Mínimo"], index=0)

        df_work = df.copy()
        if ano_sel != "Todos":
            df_work = df_work[df_work[sel_ano_col] == ano_sel]
        if uf_sel:
            df_work = df_work[df_work[sel_uf_col].isin(uf_sel)]

        # agregação
        by = []
        if sel_ano_col in df_work.columns: by.append(sel_ano_col)
        if sel_uf_col  in df_work.columns: by.append(sel_uf_col)

        if by:
            gb = df_work.groupby(by, dropna=False)[sel_metric]
            if agg_op == "Soma":    agg = gb.sum()
            elif agg_op == "Média": agg = gb.mean()
            elif agg_op == "Máximo": agg = gb.max()
            else:                   agg = gb.min()
            df_agg = agg.reset_index().rename(columns={sel_metric:"valor"})
        else:
            df_agg = df_work[[sel_metric]].rename(columns={sel_metric:"valor"})

        # taxa por 100k (se tiver UF)
        if sel_uf_col in df_agg.columns:
            df_agg["pop_mil"] = df_agg[sel_uf_col].map(POP_UF)
            df_agg["taxa_100k"] = (df_agg["valor"] / (df_agg["pop_mil"]*1000) * 100000).round(4)
        else:
            df_agg["taxa_100k"] = np.nan

        # KPIs
        k = st.columns(4)
        with k[0]:
            st.metric("Registros", f"{len(df_work):,}".replace(",","."))
        with k[1]:
            st.metric("UF distintas", f"{df_work[sel_uf_col].nunique(dropna=True)}")
        with k[2]:
            st.metric("Total (métrica)", f"{df_work[sel_metric].sum():,.0f}".replace(",","."))
        with k[3]:
            st.metric("Média/registro", f"{df_work[sel_metric].mean():,.2f}".replace(",", "."))

        st.dataframe(df_agg, use_container_width=True)
        st.download_button(
            "Baixar resumo (CSV)",
            data=df_agg.to_csv(index=False).encode("utf-8"),
            file_name="resumo_hemoprod.csv",
            mime="text/csv",
        )

        # -------- Mapa por UF (bolinhas) --------
        st.markdown("### Mapa por UF")
        if sel_uf_col in df_agg.columns:
            # pega o recorte do ano selecionado (se "Todos", pega tudo e soma por UF)
            df_map = df_agg.copy()
            if ano_sel != "Todos" and sel_ano_col in df_map.columns:
                df_map = df_map[df_map[sel_ano_col] == ano_sel]
            # agrega por UF (soma)
            df_map = df_map.groupby(sel_uf_col, dropna=True)["valor"].sum().reset_index()
            df_map.columns = ["uf", "valor"]
            df_map["valor"] = _numericize(df_map["valor"]).fillna(0)

            # lat/lon
            pts = []
            for _, r in df_map.iterrows():
                uf = str(r["uf"]).upper()
                if uf in UF_COORD:
                    lon, lat = UF_COORD[uf]
                    pts.append({"position":[lon,lat], "uf": uf, "valor": float(r["valor"])})
            df_pts = pd.DataFrame(pts)

            if not df_pts.empty:
                minv, maxv = float(df_pts["valor"].min()), float(df_pts["valor"].max())
                def _scale(v):
                    if maxv <= 0: return 5000
                    return 3000 + 12000 * ((v - minv) / max(1e-9, (maxv - minv)))

                df_pts["radius"] = df_pts["valor"].apply(_scale)
                layer = pdk.Layer(
                    "ScatterplotLayer",
                    data=df_pts,
                    get_position="position",
                    get_radius="radius",
                    pickable=True,
                    get_fill_color=[225,0,0,140],
                )
                view = pdk.ViewState(latitude=-14.2, longitude=-51.9, zoom=3.2)
                st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view,
                                         tooltip={"text":"{uf}: {valor}"}))
            else:
                st.info("Sem dados suficientes para o mapa (UF + métrica).")
        else:
            st.info("Para o mapa, é necessário ter uma coluna de UF e escolher uma métrica numérica.")

    except Exception as e:
        st.error(f"Falha ao carregar: {e}")

# =============== PÁGINA ESTADUAL (placeholder) ===============
elif page == "Estoques estaduais":
    st.subheader("Estoques por tipo sanguíneo — Fontes estaduais (upload manual)")
    st.caption("Envie um CSV com colunas: data, uf, hemocentro, tipo, rh, estoque_atual, estoque_minimo")
    up2 = st.file_uploader("Enviar CSV estadual", type=["csv"])
    if up2:
        try:
            dfe = _load_csv_robust(up2)
            dfe.columns = [c.strip().lower() for c in dfe.columns]
            if "uf" in dfe.columns:
                dfe["uf"] = _coerce_uf(dfe["uf"])
            if "estoque_atual" in dfe.columns and "estoque_minimo" in dfe.columns:
                dfe["estoque_atual"]  = _numericize(dfe["estoque_atual"])
                dfe["estoque_minimo"] = _numericize(dfe["estoque_minimo"])
                dfe["cobertura"] = (dfe["estoque_atual"] / dfe["estoque_minimo"]).replace([np.inf, -np.inf], np.nan)
            st.dataframe(dfe.head(200), use_container_width=True)
            if "cobertura" in dfe.columns:
                alertas = dfe[dfe["cobertura"] < 1.0].sort_values("cobertura")
                st.markdown("**Alertas (abaixo do mínimo)**")
                st.dataframe(alertas, use_container_width=True)
        except Exception as e:
            st.error(f"Erro ao ler CSV estadual: {e}")

# =============== SOBRE ===============
else:
    st.subheader("Sobre")
    st.markdown(
        "- Fonte nacional: **ANVISA/Hemoprod** (dados brutos).  \n"
        "- O painel usa **dados agregados** (LGPD).  \n"
        "- Código aberto para fins acadêmicos e sociais."
    )
