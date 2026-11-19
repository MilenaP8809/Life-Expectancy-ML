
# ============================================================
# LIFE EXPECTANCY ANALYTICS DASHBOARD
# WHO / KAGGLE / WORLD BANK
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests

from pathlib import Path
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# ============================================================
# 1. SIDINSTÄLLNINGAR
# ============================================================

st.set_page_config(
    page_title="Life Expectancy Analytics",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "life_data"

LOCAL_FILE = DATA_DIR / "life_expectancy_enriched.csv"
WORLD_BANK_FILE = DATA_DIR / "dashboard_worldbank.csv"

# Färgpalett
BG = "#0B1220"
PANEL = "#172235"
TEXT = "#F1F5F9"
MUTED = "#CBD5E1"
GRID = "#334155"
RED = "#EF4444"
GREEN = "#22C55E"
BLUE = "#38BDF8"


# ============================================================
# 2. DASHBOARD DESIGN
# ============================================================

st.markdown("""
<style>

.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
    max-width: 1700px;
}

[data-testid="stSidebar"] {
    background-color: #172235;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
}

[data-testid="stWidgetLabel"] {
    color: #E2E8F0;
}

.dashboard-header {
    background: #172235;
    border-left: 5px solid #EF4444;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 25px;
}

.dashboard-title {
    font-size: 30px;
    font-weight: 750;
    color: #FFFFFF;
}

.dashboard-subtitle {
    color: #CBD5E1;
    margin-top: 8px;
    font-size: 15px;
}

.kpi-card {
    background: linear-gradient(
        135deg,
        #991B1B,
        #7F1D1D
    );
    border: 1px solid #DC2626;
    padding: 18px 10px;
    border-radius: 14px;
    text-align: center;
    min-height: 112px;
}

.kpi-value {
    color: #FFFFFF;
    font-size: clamp(17px, 1.7vw, 27px);
    font-weight: 750;
    overflow-wrap: anywhere;
}

.kpi-label {
    color: #FEE2E2;
    font-size: 12px;
    margin-top: 8px;
    overflow-wrap: anywhere;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. WORLD BANK-INDIKATORER
# ============================================================

INDICATORS = {
    "Life expectancy": "SP.DYN.LE00.IN",
    "GDP_Final": "NY.GDP.PCAP.CD",
    "Population_Final": "SP.POP.TOTL",
    "Health_Expenditure": "SH.XPD.CHEX.PC.CD",
    "Infant_Mortality_Rate": "SP.DYN.IMRT.IN",
    "Measles_Vaccination": "SH.IMM.MEAS"
}

LABELS = {
    "Life expectancy": "Förväntad livslängd (år)",
    "GDP_Final": "BNP per capita (USD)",
    "Population_Final": "Befolkning",
    "Schooling": "Utbildning (år)",
    "Income composition of resources": "Inkomstrelaterat index",
    "Health_Expenditure": "Sjukvårdsutgifter (USD/invånare)",
    "Infant_Mortality_Rate": "Spädbarnsdödlighet (per 1 000)",
    "Measles_Vaccination": "Mässlingsvaccination (%)"
}

FEATURE_COLUMNS = list(LABELS.keys())


# ============================================================
# 4. LADDA HISTORISK KAGGLE-DATA
# ============================================================

@st.cache_data
def load_local_data():

    if not LOCAL_FILE.exists():
        raise FileNotFoundError(
            f"Datafilen saknas: {LOCAL_FILE}"
        )

    df = pd.read_csv(LOCAL_FILE)

    required = [
        "Country",
        "Country_Code",
        "Year"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Kolumner saknas: {missing}"
        )

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df["Country"] = (
        df["Country"]
        .astype("string")
        .str.strip()
    )

    df["Country_Code"] = (
        df["Country_Code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df = df.dropna(
        subset=required
    ).copy()

    df = df[
        df["Country_Code"].str.len() == 3
    ].copy()

    df["Year"] = df["Year"].astype(int)

    df = df.drop_duplicates(
        subset=["Country_Code", "Year"]
    )

    return df


# ============================================================
# 5. LADDA SPARAD WORLD BANK-CSV
# ============================================================

@st.cache_data
def load_worldbank_csv():

    if not WORLD_BANK_FILE.exists():
        return None

    df = pd.read_csv(WORLD_BANK_FILE)

    required = ["Country_Code", "Year"]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"World Bank-filen saknar: {missing}"
        )

    df["Country_Code"] = (
        df["Country_Code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    df["Year"] = pd.to_numeric(
        df["Year"],
        errors="coerce"
    )

    df = df.dropna(
        subset=required
    ).copy()

    df["Year"] = df["Year"].astype(int)

    df = df.drop_duplicates(
        ["Country_Code", "Year"]
    )

    return df


# ============================================================
# 6. API-UPPDATERING – VALFRI
# ============================================================

def fetch_worldbank():

    session = requests.Session()

    retry = Retry(
        total=2,
        backoff_factor=1,
        status_forcelist=[
            429, 500, 502, 503, 504
        ]
    )

    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=retry
        )
    )

    results = []

    for column, indicator in INDICATORS.items():

        url = (
            "https://api.worldbank.org/v2/"
            f"country/all/indicator/{indicator}"
        )

        params = {
            "format": "json",
            "date": "2000:2024",
            "per_page": 20000,
            "page": 1
        }

        response = session.get(
            url,
            params=params,
            timeout=45
        )

        response.raise_for_status()

        payload = response.json()

        if (
            not isinstance(payload, list)
            or len(payload) < 2
            or not isinstance(payload[1], list)
        ):
            raise ValueError(
                f"Felaktigt API-svar: {indicator}"
            )

        observations = list(payload[1])

        pages = int(
            payload[0].get("pages", 1)
        )

        for page in range(2, pages + 1):

            params["page"] = page

            response = session.get(
                url,
                params=params,
                timeout=45
            )

            response.raise_for_status()

            page_result = response.json()

            if (
                not isinstance(page_result, list)
                or len(page_result) < 2
                or not isinstance(page_result[1], list)
            ):
                raise ValueError(
                    "Ofullständigt API-svar."
                )

            observations.extend(
                page_result[1]
            )

        rows = []

        for item in observations:

            code = item.get(
                "countryiso3code"
            )

            value = item.get("value")
            year = item.get("date")

            if not code or value is None:
                continue

            rows.append({
                "Country_Code": code,
                "Year": int(year),
                column: value
            })

        part = pd.DataFrame(
            rows,
            columns=[
                "Country_Code",
                "Year",
                column
            ]
        )

        part = part.drop_duplicates(
            ["Country_Code", "Year"]
        )

        results.append(part)

    combined = results[0]

    for part in results[1:]:

        combined = combined.merge(
            part,
            on=["Country_Code", "Year"],
            how="outer",
            validate="one_to_one"
        )

    return combined


# ============================================================
# 7. KOMBINERA KAGGLE OCH WORLD BANK
# ============================================================

def prepare_data(local, worldbank):

    country_info = local[
        ["Country", "Country_Code"]
    ].drop_duplicates(
        "Country_Code"
    )

    years = pd.DataFrame({
        "Year": range(2000, 2025)
    })

    base = country_info.merge(
        years,
        how="cross"
    )

    base = base.merge(
        local,
        on=[
            "Country",
            "Country_Code",
            "Year"
        ],
        how="left",
        validate="one_to_one"
    )

    if worldbank is not None:

        base = base.merge(
            worldbank,
            on=["Country_Code", "Year"],
            how="left",
            suffixes=("", "_WB"),
            validate="one_to_one"
        )

        for col in INDICATORS.keys():

            wb_col = col + "_WB"

            if wb_col in base.columns:

                if col in base.columns:

                    base[col] = (
                        base[wb_col]
                        .combine_first(
                            base[col]
                        )
                    )

                else:
                    base[col] = base[wb_col]

                base = base.drop(
                    columns=[wb_col]
                )

    for col in FEATURE_COLUMNS:

        if col not in base.columns:
            base[col] = np.nan

        base[col] = pd.to_numeric(
            base[col],
            errors="coerce"
        )

    if "Status" not in base.columns:
        base["Status"] = pd.NA

    return base


# ============================================================
# 8. GEMENSAM DIAGRAMDESIGN
# ============================================================

def style_chart(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(
            color=TEXT,
            size=13
        ),
        margin=dict(
            l=55,
            r=35,
            t=40,
            b=55
        ),
        hoverlabel=dict(
            bgcolor="#1E293B",
            font_color="#FFFFFF"
        ),
        legend=dict(
            font=dict(color=TEXT),
            bgcolor="rgba(0,0,0,0)"
        )
    )

    fig.update_xaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
        automargin=True
    )

    fig.update_yaxes(
        gridcolor=GRID,
        zeroline=False,
        tickfont=dict(color=MUTED),
        title_font=dict(color=TEXT),
        automargin=True
    )

    return fig


def kpi_card(label, value):

    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-value">
                {value}
            </div>
            <div class="kpi-label">
                {label}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 9. LADDA DATA OCH SIDOPANEL
# ============================================================

st.sidebar.title("🌍 FILTER")

try:
    local_data = load_local_data()

except Exception as e:
    st.error(
        f"Kunde inte läsa Kaggle-data: {e}"
    )
    st.stop()


# Uppdatering är frivillig.
# Den sparade CSV-filen fungerar även offline.

if st.sidebar.button(
    "Uppdatera World Bank-data",
    use_container_width=True
):

    with st.spinner(
        "Hämtar World Bank-data..."
    ):

        try:

            new_data = fetch_worldbank()

            if new_data.empty:
                st.sidebar.warning(
                    "Ingen data hämtades."
                )

            else:

                DATA_DIR.mkdir(
                    parents=True,
                    exist_ok=True
                )

                new_data.to_csv(
                    WORLD_BANK_FILE,
                    index=False,
                    encoding="utf-8-sig"
                )

                load_worldbank_csv.clear()

                st.sidebar.success(
                    "World Bank-data har uppdaterats."
                )

        except (
            requests.RequestException,
            ValueError,
            KeyError
        ) as e:

            st.sidebar.warning(
                "API-uppdateringen misslyckades. "
                "Den lokala CSV-filen används."
            )

            st.sidebar.caption(str(e))


try:
    worldbank_data = load_worldbank_csv()

    df = prepare_data(
        local_data,
        worldbank_data
    )

except Exception as e:
    st.error(
        f"Kunde inte kombinera datakällorna: {e}"
    )
    st.stop()


if worldbank_data is not None:

    st.sidebar.success(
        "World Bank-CSV inläst"
    )

else:

    st.sidebar.warning(
        "World Bank-CSV saknas. "
        "Endast historisk Kaggle-data används."
    )


# ============================================================
# 10. FILTER
# ============================================================

year_range = st.sidebar.slider(
    "Välj tidsperiod",
    min_value=2000,
    max_value=2024,
    value=(2000, 2024)
)

countries = sorted(
    df["Country"]
    .dropna()
    .unique()
    .tolist()
)

show_all = st.sidebar.checkbox(
    "Visa alla länder",
    value=True
)

selected_countries = st.sidebar.multiselect(
    "Välj länder",
    options=countries,
    default=[
        c for c in ["Sweden", "Norway"]
        if c in countries
    ],
    disabled=show_all
)

if show_all:
    selected_countries = countries

filtered = df[
    df["Year"].between(*year_range)
    & df["Country"].isin(
        selected_countries
    )
].copy()

st.sidebar.divider()

variables = FEATURE_COLUMNS

selected_variable = st.sidebar.selectbox(
    "Variabel att visualisera",
    variables,
    format_func=lambda x: LABELS[x]
)

if filtered.empty:
    st.warning(
        "Ingen data matchar dina filter."
    )
    st.stop()


# ============================================================
# 11. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">
            🌍 LIFE EXPECTANCY ANALYTICS
        </div>
        <div class="dashboard-subtitle">
            WHO / Kaggle / World Bank |
            Interaktiv analys av livslängd,
            ekonomi och folkhälsa
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 12. KPI-KORT
# ============================================================

def latest_indicator(column):

    available = filtered.dropna(
        subset=[column]
    )

    if available.empty:
        return None, None

    year = int(
        available["Year"].max()
    )

    rows = available[
        available["Year"] == year
    ]

    value = rows[column].mean()

    return year, value


life_year, life_avg = latest_indicator(
    "Life expectancy"
)

gdp_year, gdp_avg = latest_indicator(
    "GDP_Final"
)

school_year, school_avg = latest_indicator(
    "Schooling"
)

population_year, population_avg = latest_indicator(
    "Population_Final"
)


def display_value(value, decimals=1):

    if value is None or pd.isna(value):
        return "Saknas"

    return f"{value:,.{decimals}f}"


if life_year is not None:

    latest_life = filtered[
        filtered["Year"] == life_year
    ].dropna(
        subset=["Life expectancy"]
    )

    num_countries = (
        latest_life["Country"].nunique()
    )

else:
    num_countries = 0


population_text = "Saknas"

if population_avg is not None:
    if pd.notna(population_avg):
        population_text = (
            f"{population_avg / 1_000_000:.1f} milj."
        )


k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    kpi_card(
        f"Länder ({life_year or '–'})",
        num_countries
    )

with k2:
    kpi_card(
        f"Livslängd ({life_year or '–'})",
        display_value(life_avg, 1)
    )

with k3:
    kpi_card(
        f"BNP per capita ({gdp_year or '–'})",
        display_value(gdp_avg, 0)
    )

with k4:
    kpi_card(
        f"Utbildningsår ({school_year or '–'})",
        display_value(school_avg, 1)
    )

with k5:
    kpi_card(
        f"Befolkning ({population_year or '–'})",
        population_text
    )

with k6:
    kpi_card(
        "Senaste livslängdsår",
        life_year if life_year else "Saknas"
    )


st.caption(
    "Varje KPI använder det senaste året med "
    "tillgängliga värden inom valt tidsintervall. "
    "Medelvärdena är inte befolkningsvägda. "
    "Befolkning avser genomsnitt per land."
)


# ============================================================
# 13. HISTORISK UTVECKLING
# ============================================================

st.divider()

st.subheader(
    "📈 Historisk utveckling"
)

trend_data = filtered.dropna(
    subset=[selected_variable]
)

if not trend_data.empty:

    trend = (
        trend_data
        .groupby(
            "Year",
            as_index=False
        )[selected_variable]
        .mean()
    )

    fig_trend = px.line(
        trend,
        x="Year",
        y=selected_variable,
        markers=True,
        labels={
            "Year": "År",
            selected_variable: LABELS[selected_variable]
        }
    )

    fig_trend.update_traces(
        line=dict(
            color=GREEN,
            width=4
        ),
        marker=dict(
            color=RED,
            size=9
        )
    )

    fig_trend = style_chart(
        fig_trend,
        height=400
    )

    fig_trend.update_xaxes(
        range=[
            year_range[0] - 0.2,
            year_range[1] + 0.2
        ],
        dtick=2,
        tickformat="d"
    )

    st.plotly_chart(
        fig_trend,
        use_container_width=True,
        key="trend_chart"
    )

    st.caption(
        "Diagrammet visar medelvärden av "
        "tillgängliga land-år-observationer. "
        "Antalet länder kan variera mellan åren."
    )

else:

    st.info(
        "Ingen data finns för vald variabel "
        "och tidsperiod."
    )


# ============================================================
# 14. VÄRLDSKARTA OCH RANKING
# ============================================================

st.divider()

left, right = st.columns(
    [1.65, 1],
    gap="large"
)

with left:

    st.subheader(
        "🗺️ Världskarta"
    )

    available_map_years = sorted(
        filtered.loc[
            filtered[selected_variable].notna(),
            "Year"
        ].unique()
    )

    if available_map_years:

        map_year = st.select_slider(
            "År på kartan",
            options=available_map_years,
            value=available_map_years[-1]
        )

        map_data = filtered[
            filtered["Year"] == map_year
        ].dropna(
            subset=[
                selected_variable,
                "Country_Code"
            ]
        ).copy()

        fig_map = px.scatter_geo(
            map_data,
            locations="Country_Code",
            locationmode="ISO-3",
            hover_name="Country",
            color=selected_variable,
            color_continuous_scale="RdYlGn",
            projection="natural earth",
            labels={
                selected_variable: LABELS[selected_variable]
            },
            hover_data={
                "Year": True,
                "Country_Code": False
            }
        )

        fig_map.update_traces(
            marker=dict(
                size=10,
                opacity=0.9,
                line=dict(
                    color="#FFFFFF",
                    width=0.5
                )
            )
        )

        fig_map.update_geos(
            bgcolor=PANEL,
            showframe=False,
            showland=True,
            landcolor="#334155",
            showocean=True,
            oceancolor="#101827",
            showcoastlines=True,
            coastlinecolor="#64748B",
            showcountries=True,
            countrycolor="#64748B"
        )

        fig_map.update_layout(
            template="plotly_dark",
            height=470,
            paper_bgcolor=PANEL,
            font=dict(color=TEXT),
            margin=dict(
                l=0,
                r=0,
                t=20,
                b=0
            ),
            coloraxis_colorbar=dict(
                title=dict(
                    text=LABELS[selected_variable],
                    font=dict(color=TEXT)
                ),
                tickfont=dict(color=TEXT),
                thickness=12,
                len=0.7
            )
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True,
            key="world_map"
        )

    else:

        map_year = None

        st.info(
            "Ingen kartdata finns "
            "för den valda variabeln."
        )


with right:

    st.subheader(
        "🏆 Länder – Top 5 / Bottom 5"
    )

    rank_mode = st.radio(
        "Välj ranking",
        [
            "Bästa 5",
            "Sämsta 5"
        ],
        horizontal=True
    )

    if map_year is not None:

        rank_data = filtered[
            filtered["Year"] == map_year
        ].dropna(
            subset=[selected_variable]
        )[
            ["Country", selected_variable]
        ].copy()

        ascending = (
            rank_mode == "Sämsta 5"
        )

        rank_data = rank_data.sort_values(
            selected_variable,
            ascending=ascending
        ).head(5)

        if not rank_data.empty:

            rank_data = rank_data.sort_values(
                selected_variable,
                ascending=True
            )

            fig_rank = px.bar(
                rank_data,
                x=selected_variable,
                y="Country",
                orientation="h",
                text=selected_variable,
                labels={
                    selected_variable: LABELS[selected_variable],
                    "Country": ""
                }
            )

            fig_rank.update_traces(
                marker_color=(
                    GREEN
                    if not ascending
                    else RED
                ),
                texttemplate="%{text:,.1f}",
                textposition="outside",
                cliponaxis=False,
                textfont=dict(
                    color=TEXT,
                    size=12
                )
            )

            fig_rank = style_chart(
                fig_rank,
                height=470
            )

            maximum = float(
                rank_data[selected_variable].max()
            )

            minimum = float(
                rank_data[selected_variable].min()
            )

            if minimum >= 0:
                lower_limit = 0
                upper_limit = (
                    maximum * 1.2
                    if maximum > 0
                    else 1
                )

                fig_rank.update_xaxes(
                    range=[
                        lower_limit,
                        upper_limit
                    ]
                )

            fig_rank.update_layout(
                showlegend=False,
                margin=dict(
                    l=15,
                    r=70,
                    t=35,
                    b=55
                )
            )

            st.plotly_chart(
                fig_rank,
                use_container_width=True,
                key="ranking_chart"
            )

        else:

            st.info(
                "Ingen ranking tillgänglig."
            )

    else:

        st.info(
            "Välj en variabel med data."
        )


# ============================================================
# 15. INTERAKTIV SAMBANDSANALYS
# ============================================================

st.divider()

st.subheader(
    "🔎 Interaktiv sambandsanalys"
)

st.write(
    "Undersök sambandet mellan två variabler. "
    "Välj exempelvis BNP per capita och livslängd."
)


# ------------------------------------------------------------
# 15.1 VÄLJ VARIABLER
# ------------------------------------------------------------

col_x, col_y = st.columns(2)

with col_x:

    x_var = st.selectbox(
        "Välj X-variabel",
        variables,
        index=1,
        format_func=lambda x: LABELS[x],
        key="scatter_x"
    )

with col_y:

    y_var = st.selectbox(
        "Välj Y-variabel",
        variables,
        index=0,
        format_func=lambda x: LABELS[x],
        key="scatter_y"
    )


# ------------------------------------------------------------
# 15.2 KONTROLLERA VARIABLER
# ------------------------------------------------------------

if x_var == y_var:

    st.warning(
        "Välj två olika variabler."
    )

else:

    scatter_all = filtered.dropna(
        subset=[
            x_var,
            y_var,
            "Country",
            "Year"
        ]
    ).copy()

    if scatter_all.empty:

        st.warning(
            "Ingen data finns för denna kombination."
        )

    else:

        # ----------------------------------------------------
        # 15.3 VÄLJ ANALYSPERIOD
        # ----------------------------------------------------

        analysis_mode = st.radio(
            "Analysperiod",
            [
                "Ett år",
                "Alla valda år"
            ],
            horizontal=True,
            key="scatter_period"
        )

        available_years = sorted(
            scatter_all["Year"].unique()
        )

        if analysis_mode == "Ett år":

            scatter_year = st.select_slider(
                "Välj år för jämförelsen",
                options=available_years,
                value=available_years[-1],
                key="scatter_year"
            )

            scatter_data = scatter_all[
                scatter_all["Year"] == scatter_year
            ].copy()

        else:

            scatter_data = scatter_all.copy()


        # ----------------------------------------------------
        # 15.4 LOGARITMISK X-SKALA
        # ----------------------------------------------------

        use_log = st.checkbox(
            "Använd logaritmisk X-skala",
            value=(x_var == "GDP_Final"),
            key="scatter_log"
        )

        if use_log:

            scatter_data = scatter_data[
                scatter_data[x_var] > 0
            ].copy()


        if scatter_data.empty:

            st.warning(
                "Ingen data finns med positiva X-värden."
            )

        else:

            # ------------------------------------------------
            # 15.5 PEARSON-KORRELATION
            # ------------------------------------------------

            pearson = scatter_data[
                [x_var, y_var]
            ].corr(
                method="pearson"
            ).iloc[0, 1]

            st.markdown(
                "### Statistisk sammanfattning"
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric(
                    "Antal observationer",
                    len(scatter_data)
                )

            with m2:
                st.metric(
                    "Antal länder",
                    scatter_data["Country"].nunique()
                )

            with m3:
                st.metric(
                    "Pearson-korrelation (r)",
                    (
                        f"{pearson:.3f}"
                        if pd.notna(pearson)
                        else "Saknas"
                    )
                )


            # ------------------------------------------------
            # 15.6 FÄRGKODNING
            # ------------------------------------------------

            if (
                "Status" in scatter_data.columns
                and scatter_data["Status"].notna().any()
            ):

                color_column = "Status"

                color_map = {
                    "Developed": BLUE,
                    "Developing": GREEN
                }

            else:

                color_column = None
                color_map = None


            # ------------------------------------------------
            # 15.7 SPRIDNINGSDIAGRAM
            # ------------------------------------------------

            fig_scatter = px.scatter(
                scatter_data,
                x=x_var,
                y=y_var,
                color=color_column,
                color_discrete_map=color_map,
                hover_name="Country",
                hover_data={
                    "Year": True,
                    x_var: ":,.2f",
                    y_var: ":,.2f"
                },
                labels={
                    x_var: LABELS[x_var],
                    y_var: LABELS[y_var],
                    "Status": "Utvecklingsstatus"
                },
                opacity=0.75
            )

            fig_scatter.update_traces(
                marker=dict(
                    size=9,
                    line=dict(
                        width=0.5,
                        color="#E2E8F0"
                    )
                )
            )

            if color_column is None:

                fig_scatter.update_traces(
                    marker_color=BLUE
                )


            # ------------------------------------------------
            # 15.8 TRENDLINJE
            # ------------------------------------------------

            show_trendline = st.checkbox(
                "Visa trendlinje",
                value=True,
                key="scatter_trendline"
            )

            if (
                show_trendline
                and len(scatter_data) >= 3
                and scatter_data[x_var].nunique() > 1
            ):

                x_values = scatter_data[
                    x_var
                ].to_numpy(
                    dtype=float
                )

                y_values = scatter_data[
                    y_var
                ].to_numpy(
                    dtype=float
                )

                if use_log:

                    regression_x = np.log10(
                        x_values
                    )

                    line_x = np.geomspace(
                        x_values.min(),
                        x_values.max(),
                        100
                    )

                    line_regression_x = np.log10(
                        line_x
                    )

                else:

                    regression_x = x_values

                    line_x = np.linspace(
                        x_values.min(),
                        x_values.max(),
                        100
                    )

                    line_regression_x = line_x

                coefficients = np.polyfit(
                    regression_x,
                    y_values,
                    1
                )

                line_y = np.polyval(
                    coefficients,
                    line_regression_x
                )

                fig_scatter.add_trace(
                    go.Scatter(
                        x=line_x,
                        y=line_y,
                        mode="lines",
                        name="Trendlinje",
                        line=dict(
                            color="#F87171",
                            width=3,
                            dash="dash"
                        ),
                        hoverinfo="skip"
                    )
                )


            # ------------------------------------------------
            # 15.9 GEMENSAM DESIGN
            # ------------------------------------------------

            fig_scatter = style_chart(
                fig_scatter,
                height=540
            )

            fig_scatter.update_layout(
                font=dict(
                    color=TEXT,
                    size=13
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="left",
                    x=0
                ),
                margin=dict(
                    l=65,
                    r=35,
                    t=70,
                    b=65
                )
            )


            # ------------------------------------------------
            # 15.10 TYDLIG BNP-AXEL
            # ------------------------------------------------

            if use_log and x_var == "GDP_Final":

                fig_scatter.update_xaxes(
                    type="log",
                    title_text="BNP per capita (USD)",
                    tickmode="array",
                    tickvals=[
                        100,
                        500,
                        1000,
                        5000,
                        10000,
                        50000,
                        100000,
                        500000
                    ],
                    ticktext=[
                        "100",
                        "500",
                        "1 000",
                        "5 000",
                        "10 000",
                        "50 000",
                        "100 000",
                        "500 000"
                    ],
                    gridcolor=GRID,
                    tickfont=dict(
                        color=MUTED,
                        size=12
                    ),
                    title_font=dict(
                        color=TEXT
                    )
                )

            else:

                fig_scatter.update_xaxes(
                    type=(
                        "log"
                        if use_log
                        else "linear"
                    ),
                    title_text=LABELS[x_var],
                    gridcolor=GRID,
                    tickfont=dict(
                        color=MUTED
                    ),
                    title_font=dict(
                        color=TEXT
                    )
                )


            # ------------------------------------------------
            # 15.11 Y-AXEL
            # ------------------------------------------------

            fig_scatter.update_yaxes(
                title_text=LABELS[y_var],
                gridcolor=GRID,
                tickfont=dict(
                    color=MUTED
                ),
                title_font=dict(
                    color=TEXT
                )
            )


            # ------------------------------------------------
            # 15.12 VISA DIAGRAM
            # ------------------------------------------------

            st.plotly_chart(
                fig_scatter,
                use_container_width=True,
                key="scatter_chart_v2"
            )


            # ------------------------------------------------
            # 15.13 TOLKNING AV KORRELATION
            # ------------------------------------------------

            st.markdown(
                "### Tolkning av sambandet"
            )

            if pd.isna(pearson):

                st.write(
                    "Korrelation kunde inte beräknas."
                )

            else:

                if abs(pearson) >= 0.7:
                    interpretation = "starkt"

                elif abs(pearson) >= 0.4:
                    interpretation = "måttligt"

                elif abs(pearson) >= 0.2:
                    interpretation = "svagt"

                else:
                    interpretation = "mycket svagt"

                if pearson > 0:
                    direction = "positivt"

                elif pearson < 0:
                    direction = "negativt"

                else:
                    direction = "neutralt"

                st.write(
                    f"Pearson r = {pearson:.3f}. "
                    f"Resultatet visar ett {interpretation} "
                    f"{direction} linjärt samband mellan "
                    f"{LABELS[x_var]} och "
                    f"{LABELS[y_var]} "
                    "i det valda urvalet."
                )

            st.caption(
                "Pearson-korrelationen beräknas på "
                "de ursprungliga värdena. "
                "Vid logaritmisk X-skala anpassas "
                "trendlinjen efter log10(X). "
                "Vid analys av alla år kan samma land "
                "förekomma flera gånger. "
                "Korrelation innebär inte orsakssamband."
            )


# ============================================================
# 16. FILTRERAD DATATABELL
# ============================================================

st.divider()

st.subheader(
    "📋 Filtrerad datatabell"
)

main_columns = [
    "Country",
    "Year"
] + FEATURE_COLUMNS

table = filtered[
    main_columns
].sort_values(
    ["Year", "Country"],
    ascending=[False, True]
)

st.dataframe(
    table,
    use_container_width=True,
    hide_index=True
)


# ============================================================
# 17. LADDA NER FILTRERAD DATA
# ============================================================

st.download_button(
    "Ladda ner filtrerad data (CSV)",
    data=table.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="life_expectancy_dashboard.csv",
    mime="text/csv"
)


# ============================================================
# SLUT PÅ DASHBOARDEN
# ============================================================
