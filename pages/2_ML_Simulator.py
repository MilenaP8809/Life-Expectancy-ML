
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import joblib

from pathlib import Path


# ======================================
# 1. INSTÄLLNINGAR
# ======================================

st.set_page_config(
    page_title="ML Simulator",
    page_icon="🧠",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR / "life_data"
    / "life_expectancy_enriched.csv"
)

MODEL_PATH = (
    BASE_DIR / "models"
    / "life_expectancy_rf.joblib"
)

FEATURES = [
    "Schooling",
    "Income composition of resources",
    "GDP_Final",
    "Population_Final"
]

LABELS = {
    "Schooling": "Utbildning (år)",
    "Income composition of resources": "Inkomstrelaterat index",
    "GDP_Final": "BNP per capita (USD)",
    "Population_Final": "Befolkning"
}

BG = "#0B1220"
PANEL = "#172235"
TEXT = "#F1F5F9"
MUTED = "#CBD5E1"
RED = "#EF4444"
GREEN = "#22C55E"
BLUE = "#38BDF8"


# ======================================
# 2. DESIGN
# ======================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

.block-container {
    max-width: 1600px;
    padding-top: 1.5rem;
}

[data-testid="stSidebar"] {
    background-color: #172235;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
}

.sim-header {
    background: #172235;
    border-left: 5px solid #EF4444;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
}

.sim-title {
    font-size: 30px;
    font-weight: 750;
    color: white;
}

.sim-subtitle {
    color: #CBD5E1;
    margin-top: 7px;
}

.result-box {
    background: #172235;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 22px;
    text-align: center;
}

.result-value {
    font-size: 36px;
    font-weight: 750;
    color: #FFFFFF;
}

.result-label {
    font-size: 15px;
    color: #CBD5E1;
}
</style>
""", unsafe_allow_html=True)


# ======================================
# 3. LADDA MODELL OCH DATA
# ======================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():

    df = pd.read_csv(DATA_PATH)

    df["Year"] = pd.to_numeric(
        df["Year"], errors="coerce"
    )

    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col], errors="coerce"
        )

    return df


if not MODEL_PATH.exists():
    st.error(
        f"Modellen saknas: {MODEL_PATH}"
    )
    st.stop()

if not DATA_PATH.exists():
    st.error(
        f"Datafilen saknas: {DATA_PATH}"
    )
    st.stop()


try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error(
        f"Kunde inte läsa modellen eller data: "
        f"{type(e).__name__}: {e}"
    )
    st.exception(e)
    st.stop()



# ======================================
# 4. PREDIKTIONSFUNKTION
# ======================================

def predict_life(values):

    X = pd.DataFrame(
        [values],
        columns=FEATURES
    )

    prediction = model.predict(X)

    return float(prediction[0])


# ======================================
# 5. RUBRIK
# ======================================

st.markdown("""
<div class="sim-header">
    <div class="sim-title">
        🧠 ML PROGNOS & SIMULERING
    </div>
    <div class="sim-subtitle">
        Random Forest |
        Förväntad livslängd |
        Scenarioanalys
    </div>
</div>
""", unsafe_allow_html=True)

st.info(
    "Modellen uppskattar livslängd utifrån fyra "
    "variabler. Detta är en scenarioprediktion, "
    "inte en validerad framtidsprognos."
)


# ======================================
# 6. VÄLJ LAND OCH ÅR
# ======================================

st.subheader("🌍 Välj utgångspunkt")

# Endast observationer med kompletta modellvariabler
complete = df.dropna(
    subset=["Country", "Year"] + FEATURES
).copy()

if complete.empty:
    st.error(
        "Ingen historisk observation har "
        "alla fyra modellvariablerna."
    )
    st.stop()

countries = sorted(
    complete["Country"].unique().tolist()
)

default_country = (
    countries.index("Sweden")
    if "Sweden" in countries
    else 0
)

c1, c2 = st.columns(2)

with c1:
    country = st.selectbox(
        "Välj land",
        countries,
        index=default_country
    )

country_df = complete[
    complete["Country"] == country
].copy()

years = sorted(
    country_df["Year"].astype(int).unique(),
    reverse=True
)

with c2:
    year = st.selectbox(
        "Välj historiskt år",
        years
    )

source_row = country_df[
    country_df["Year"] == year
].iloc[0]

baseline = {
    col: float(source_row[col])
    for col in FEATURES
}

st.caption(
    f"Utgångsvärdena kommer från den sparade "
    f"historiska observationen för {country}, {year}."
)


# ======================================
# 7. TVÅ SCENARIER
# ======================================

st.divider()
st.subheader("🎛️ Anpassa dina scenarier")

left, right = st.columns(2, gap="large")


def scenario_inputs(prefix, defaults):

    values = {}

    st.markdown(f"### Scenario {prefix}")

    values["Schooling"] = st.slider(
        "Utbildning (år)",
        min_value=0.0,
        max_value=25.0,
        value=float(np.clip(
            defaults["Schooling"], 0, 25
        )),
        step=0.5,
        key=f"{prefix}_school"
    )

    values["Income composition of resources"] = st.slider(
        "Inkomstrelaterat index",
        min_value=0.0,
        max_value=1.0,
        value=float(np.clip(
            defaults["Income composition of resources"],
            0, 1
        )),
        step=0.01,
        key=f"{prefix}_income"
    )

    values["GDP_Final"] = st.number_input(
        "BNP per capita (USD)",
        min_value=0.0,
        max_value=250000.0,
        value=float(np.clip(
            defaults["GDP_Final"], 0, 250000
        )),
        step=500.0,
        key=f"{prefix}_gdp"
    )

    values["Population_Final"] = st.number_input(
        "Befolkning",
        min_value=1.0,
        max_value=2000000000.0,
        value=float(np.clip(
            defaults["Population_Final"],
            1, 2000000000
        )),
        step=100000.0,
        key=f"{prefix}_pop"
    )

    return values


with left:
    scenario_a = scenario_inputs(
        "A", baseline
    )

with right:
    scenario_b = scenario_inputs(
        "B", baseline
    )


# ======================================
# 8. BERÄKNA PREDIKTIONER
# ======================================

try:
    prediction_a = predict_life(scenario_a)
    prediction_b = predict_life(scenario_b)

except Exception as e:
    st.error(
        f"Kunde inte läsa modellen eller data: "
        f"{type(e).__name__}: {e}"
    )
    st.exception(e)
    st.stop()


  


difference = prediction_b - prediction_a

st.divider()
st.subheader("📊 Resultat från Random Forest")

r1, r2, r3 = st.columns(3)

with r1:
    st.markdown(
        f"""
        <div class="result-box">
          <div class="result-value">
            {prediction_a:.1f} år
          </div>
          <div class="result-label">
            Scenario A
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with r2:
    st.markdown(
        f"""
        <div class="result-box">
          <div class="result-value">
            {prediction_b:.1f} år
          </div>
          <div class="result-label">
            Scenario B
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with r3:
    sign = "+" if difference > 0 else ""
    st.markdown(
        f"""
        <div class="result-box">
          <div class="result-value">
            {sign}{difference:.1f} år
          </div>
          <div class="result-label">
            Skillnad B minus A
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================
# 9. JÄMFÖRELSEDIAGRAM
# ======================================

fig_compare = go.Figure()

fig_compare.add_trace(
    go.Bar(
        x=["Scenario A", "Scenario B"],
        y=[prediction_a, prediction_b],
        marker_color=[BLUE, GREEN],
        text=[
            f"{prediction_a:.1f} år",
            f"{prediction_b:.1f} år"
        ],
        textposition="outside",
        textfont=dict(
            size=15,
            color=TEXT
        )
    )
)

fig_compare.update_layout(
    template="plotly_dark",
    paper_bgcolor=PANEL,
    plot_bgcolor=PANEL,
    height=360,
    font=dict(color=TEXT),
    yaxis=dict(
        title="Predikterad livslängd (år)",
        range=[
            0,
            max(prediction_a, prediction_b, 1) * 1.18
        ],
        gridcolor="#334155"
    ),
    xaxis=dict(
        title="",
        tickfont=dict(color=TEXT, size=13)
    ),
    margin=dict(
        l=40, r=35, t=35, b=40
    ),
    showlegend=False
)

st.plotly_chart(
    fig_compare,
    use_container_width=True,
    key="scenario_comparison"
)


# ======================================
# 10. KÄNSLIGHETSANALYS
# ======================================

st.divider()
st.subheader("📈 Hur påverkas modellens prediktion?")

st.write(
    "Välj en variabel. Vi ändrar den inom "
    "ett observerat intervall i datasetet, "
    "medan övriga variabler hålls fasta "
    "enligt Scenario A."
)

selected_feature = st.selectbox(
    "Variabel att undersöka",
    FEATURES,
    format_func=lambda x: LABELS[x]
)

historical_values = complete[
    selected_feature
].dropna()

low = float(historical_values.quantile(0.05))
high = float(historical_values.quantile(0.95))

if high <= low:
    st.warning(
        "Variabeln saknar tillräcklig variation."
    )
else:

    values = np.linspace(
        low, high, 35
    )

    predictions = []

    for value in values:

        test_scenario = scenario_a.copy()
        test_scenario[selected_feature] = float(value)

        predictions.append(
            predict_life(test_scenario)
        )

    fig_sensitivity = go.Figure()

    fig_sensitivity.add_trace(
        go.Scatter(
            x=values,
            y=predictions,
            mode="lines+markers",
            line=dict(
                color=GREEN,
                width=3
            ),
            marker=dict(
                size=5,
                color=BLUE
            ),
            name="Modellprediktion"
        )
    )

    fig_sensitivity.add_vline(
        x=scenario_a[selected_feature],
        line_color=RED,
        line_dash="dash"
    )

    fig_sensitivity.update_layout(
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        height=420,
        font=dict(color=TEXT, size=13),
        xaxis=dict(
            title=LABELS[selected_feature],
            gridcolor="#334155"
        ),
        yaxis=dict(
            title="Predikterad livslängd (år)",
            gridcolor="#334155"
        ),
        margin=dict(
            l=50, r=30, t=30, b=50
        )
    )

    st.plotly_chart(
        fig_sensitivity,
        use_container_width=True,
        key="sensitivity_analysis"
    )

    st.caption(
        "Kurvan visar modellens respons när "
        "en variabel ändras. Den bevisar inte "
        "ett orsakssamband. Kombinationer av "
        "variabler kan också vara orealistiska."
    )


# ======================================
# 11. HISTORISK JÄMFÖRELSE
# ======================================

st.divider()
st.subheader("🔍 Historisk jämförelse")

if "Life expectancy" in source_row.index:

    actual = pd.to_numeric(
        pd.Series([source_row["Life expectancy"]]),
        errors="coerce"
    ).iloc[0]

    if pd.notna(actual):

        historical_prediction = predict_life(baseline)

        a, b = st.columns(2)

        with a:
            st.metric(
                "Observerad livslängd",
                f"{actual:.1f} år"
            )

        with b:
            st.metric(
                "Modellens uppskattning",
                f"{historical_prediction:.1f} år"
            )

        st.caption(
            "Detta är en illustration för den "
            "valda historiska observationen, "
            "inte ett oberoende testresultat. "
            "Observationen kan ha ingått i "
            "modellens träning."
        )

    else:
        st.info(
            "Observerad livslängd saknas "
            "för valt land och år."
        )


# ======================================
# 12. OBSERVERADE INTERVALL
# ======================================

st.divider()

with st.expander("Datakvalitet och modellbegränsningar"):

    st.write(
        "Modellen är tränad med historiska "
        "data. Prediktioner utanför de "
        "observerade intervallen är osäkra."
    )

    for feature in FEATURES:

        observed_min = complete[feature].min()
        observed_max = complete[feature].max()

        outside_a = not (
            observed_min <= scenario_a[feature] <= observed_max
        )

        outside_b = not (
            observed_min <= scenario_b[feature] <= observed_max
        )

        if outside_a or outside_b:
            st.warning(
                f"{LABELS[feature]}: minst ett scenario "
                "ligger utanför observerat intervall "
                f"({observed_min:,.2f}–{observed_max:,.2f})."
            )

    st.write(
        "Resultaten ska tolkas som modellbaserade "
        "scenarier, inte som säkra prognoser "
        "för ett visst framtida år."
    )
