
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# 1. INSTÄLLNINGAR
# ==========================================

st.set_page_config(
    page_title="Framtidsprognos 2036",
    page_icon="🔮",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent

WORLD_BANK = ROOT / "life_data" / "dashboard_worldbank.csv"
KAGGLE = ROOT / "life_data" / "life_expectancy_enriched.csv"

DARK = "#0B1220"
PANEL = "#172235"
WHITE = "#F1F5F9"
SOFT = "#CBD5E1"
GREEN = "#22C55E"
RED = "#F87171"
BLUE = "#38BDF8"


# ==========================================
# 2. DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #0B1220;
    color: #F1F5F9;
}

[data-testid="stSidebar"] {
    background-color: #172235;
}

.block-container {
    padding-top: 1.5rem;
    max-width: 1600px;
}

h1, h2, h3 {
    color: #F1F5F9 !important;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. VARIABLER
# ==========================================

DATA_COLUMNS = {
    "GDP_Final": "BNP per capita",
    "Health_Expenditure": "Sjukvårdsutgifter per capita",
    "Population_Final": "Befolkning"
}


# ==========================================
# 4. LADDA DATA
# ==========================================

@st.cache_data
def load_data():

    if not WORLD_BANK.exists() or not KAGGLE.exists():
        raise FileNotFoundError(
            "Både dashboard_worldbank.csv och "
            "life_expectancy_enriched.csv måste finnas i life_data."
        )

    wb = pd.read_csv(WORLD_BANK)

    who = pd.read_csv(
        KAGGLE,
        usecols=["Country", "Country_Code"]
    )

    for required in [
        "Country_Code",
        "Year",
        "Life expectancy"
    ]:
        if required not in wb.columns:
            raise ValueError(
                f"Kolumnen {required} saknas i World Bank-filen."
            )

    wb["Country_Code"] = (
        wb["Country_Code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    who["Country_Code"] = (
        who["Country_Code"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    names = (
        who.dropna(subset=["Country", "Country_Code"])
        .drop_duplicates("Country_Code")
    )

    data = wb.merge(
        names,
        on="Country_Code",
        how="inner",
        validate="many_to_one"
    )

    for col in [
        "Year",
        "Life expectancy",
        *DATA_COLUMNS
    ]:
        if col not in data.columns:
            data[col] = np.nan

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    data = data.dropna(subset=["Year"]).copy()
    data["Year"] = data["Year"].astype(int)

    data = data[
        data["Year"].between(2000, 2024)
    ]

    data = data.sort_values(["Country", "Year"])

    return data.drop_duplicates(
        ["Country_Code", "Year"]
    )


# ==========================================
# 5. GEMENSAM DIAGRAMDESIGN
# ==========================================

def design_plot(fig, height=480):

    fig.update_layout(
        height=height,
        template="plotly_dark",
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=dict(color=WHITE, size=13),
        margin=dict(l=55, r=35, t=40, b=55),
        hovermode="x unified",
        xaxis=dict(
            title="År",
            gridcolor="#334155",
            tickformat="d"
        ),
        yaxis=dict(
            title="Förväntad livslängd (år)",
            gridcolor="#334155"
        ),
        legend=dict(
            orientation="h",
            y=1.12,
            x=0
        )
    )

    return fig


# ==========================================
# 6. FÖRBERED MODELLVARIABLER
# ==========================================

def transform_features(frame, selection):

    x = pd.DataFrame(index=frame.index)

    x["År"] = frame["Year"].astype(float) - 2000.0

    for col in selection:
        x[col] = np.log1p(
            frame[col].astype(float)
        )

    return x


# ==========================================
# 7. TRÄNA PROGNOSMODELL
# ==========================================

def fitted_model(history, selection):

    required = [
        "Year",
        "Life expectancy",
        *selection
    ]

    complete = history.dropna(
        subset=required
    ).copy()

    for col in selection:
        complete = complete[
            complete[col] >= 0
        ]

    if (
        len(complete) < 10
        or complete["Year"].nunique() < 10
    ):
        raise ValueError(
            "För få kompletta historiska år. "
            "Avmarkera någon variabel eller välj "
            "ett annat land. Minst 10 år krävs."
        )

    estimator = make_pipeline(
        StandardScaler(),
        Ridge(alpha=10.0)
    )

    estimator.fit(
        transform_features(complete, selection),
        complete["Life expectancy"].astype(float)
    )

    return estimator, complete


# ==========================================
# 8. BERÄKNA FRAMTIDA VARIABELVÄRDEN
# ==========================================

def project_one(history, year, selection, growth_rates):

    row = {"Year": float(year)}

    for col in selection:

        available = history.dropna(
            subset=[col]
        ).sort_values("Year")

        available = available[
            available[col] >= 0
        ]

        if available.empty:
            raise ValueError(
                f"Inga värden finns för {DATA_COLUMNS[col]}."
            )

        latest = available.iloc[-1]

        last_val = float(latest[col])
        last_year = int(latest["Year"])

        annual = growth_rates[col] / 100.0

        row[col] = (
            last_val *
            (1.0 + annual) ** max(0, year - last_year)
        )

    return pd.DataFrame([row])


# ==========================================
# 9. HISTORISKA TILLVÄXTTAKTER
# ==========================================

def growth_from_history(history, selection):

    result = {}

    for col in selection:

        valid = history.dropna(
            subset=[col]
        ).sort_values("Year")

        valid = valid[
            valid[col] > 0
        ].tail(4)

        if len(valid) < 2:
            result[col] = 0.0
            continue

        years = int(
            valid["Year"].iloc[-1] -
            valid["Year"].iloc[0]
        )

        if years <= 0:
            result[col] = 0.0
            continue

        a = float(valid[col].iloc[0])
        b = float(valid[col].iloc[-1])

        annual = (b / a) ** (1 / years) - 1

        result[col] = float(
            np.clip(annual * 100, -10, 12)
        )

    return result


# ==========================================
# 10. HISTORISK VALIDERING
# ==========================================

def walk_forward_backtest(history, selection):

    target_years = sorted(
        history.loc[
            history["Life expectancy"].notna(),
            "Year"
        ].unique()
    )

    eligible = target_years[-6:]

    scores = []

    for year in eligible:

        training = history[
            history["Year"] < year
        ].copy()

        actual_rows = history[
            (history["Year"] == year) &
            history["Life expectancy"].notna()
        ]

        if actual_rows.empty:
            continue

        try:

            model, complete = fitted_model(
                training,
                selection
            )

            growth = growth_from_history(
                training,
                selection
            )

            projected = project_one(
                training,
                int(year),
                selection,
                growth
            )

            forecast_raw = float(
                model.predict(
                    transform_features(projected, selection)
                )[0]
            )

            last_life = (
                training
                .dropna(subset=["Life expectancy"])
                .sort_values("Year")
                .iloc[-1]
            )

            anchor_year = int(last_life["Year"])

            anchor_input = project_one(
                training,
                anchor_year,
                selection,
                growth
            )

            anchor_raw = float(
                model.predict(
                    transform_features(anchor_input, selection)
                )[0]
            )

            pred = (
                float(last_life["Life expectancy"]) +
                forecast_raw -
                anchor_raw
            )

            actual = float(
                actual_rows["Life expectancy"].iloc[0]
            )

            scores.append({
                "År": int(year),
                "Faktiskt": actual,
                "Prognos": pred,
                "Fel": pred - actual
            })

        except (ValueError, KeyError):
            continue

    return pd.DataFrame(scores)


# ==========================================
# 11. LADDA DATA
# ==========================================

try:
    data = load_data()

except (FileNotFoundError, ValueError) as err:
    st.error(str(err))
    st.stop()


# ==========================================
# 12. RUBRIK
# ==========================================

st.title(
    "🔮 Framtidsprognos för livslängd · 2025–2036"
)

st.caption(
    "World Bank-data lagrad lokalt | "
    "Separat prognosmodell | "
    "Din Random Forest-modell påverkas inte"
)

st.info(
    "Detta är en **utforskande villkorad prognos**, "
    "inte en säker framtidsbedömning. "
    "Samband med BNP, vårdutgifter och befolkning "
    "är inte automatiskt orsakssamband."
)


# ==========================================
# 13. VÄLJ LAND OCH PROGNOSÅR
# ==========================================

countries = sorted(
    data["Country"].dropna().unique()
)

if not countries:
    st.error(
        "Det finns inga matchande landsnamn "
        "mellan de två CSV-filerna."
    )
    st.stop()

left, right = st.columns(2)

with left:

    default_idx = (
        countries.index("Sweden")
        if "Sweden" in countries
        else 0
    )

    country = st.selectbox(
        "Välj land",
        countries,
        index=default_idx
    )

with right:

    end_year = st.slider(
        "Prognos till och med",
        min_value=2025,
        max_value=2036,
        value=2036
    )


country_data = data[
    data["Country"] == country
].sort_values("Year").copy()

observed = country_data.dropna(
    subset=["Life expectancy"]
)

if observed.empty:
    st.warning(
        "Inga historiska livslängdsuppgifter "
        "för detta land."
    )
    st.stop()

latest_actual_year = int(
    observed["Year"].max()
)

latest_actual = float(
    observed.iloc[-1]["Life expectancy"]
)

st.caption(
    f"Senast observerad livslängd: "
    f"{latest_actual:.2f} år ({latest_actual_year})."
)


# ==========================================
# 14. VÄLJ FÖRKLARINGSVARIABLER
# ==========================================

st.subheader(
    "1. Välj förklaringsvariabler"
)

selection = st.multiselect(
    "Förklara trenden med",
    options=list(DATA_COLUMNS),
    default=["GDP_Final"],
    format_func=lambda s: DATA_COLUMNS[s]
)

st.caption(
    "Årtal ingår alltid i prognosmodellen. "
    "De andra variablerna kan du välja själv."
)


# ==========================================
# 15. ANTAGANDEN FÖR FRAMTIDEN
# ==========================================

rates = {}

if selection:

    st.subheader(
        "2. Anta årlig förändring"
    )

    cols = st.columns(len(selection))

    defaults = {
        "GDP_Final": 2.0,
        "Health_Expenditure": 2.0,
        "Population_Final": 0.3
    }

    for i, col in enumerate(selection):

        with cols[i]:

            rates[col] = st.slider(
                f"{DATA_COLUMNS[col]} (%/år)",
                min_value=-5.0,
                max_value=10.0,
                value=defaults[col],
                step=0.5,
                key=f"rate_{col}"
            )

    st.caption(
        "Tillväxttalen är dina antaganden "
        "och ska inte tolkas som officiella "
        "World Bank-prognoser."
    )


# ==========================================
# 16. BERÄKNA PROGNOSEN
# ==========================================

try:

    model, train = fitted_model(
        country_data,
        selection
    )

    forecast_years = np.arange(
        latest_actual_year + 1,
        end_year + 1
    )

    if len(forecast_years) == 0:
        st.warning(
            "Valt slutår är inte efter "
            "senaste observerade år."
        )
        st.stop()

    projected_rows = [
        project_one(
            country_data,
            int(y),
            selection,
            rates
        )
        for y in forecast_years
    ]

    future_x = pd.concat(
        projected_rows,
        ignore_index=True
    )

    anchor_input = project_one(
        country_data,
        latest_actual_year,
        selection,
        rates
    )

    anchor_prediction = float(
        model.predict(
            transform_features(
                anchor_input,
                selection
            )
        )[0]
    )

    future_raw = model.predict(
        transform_features(
            future_x,
            selection
        )
    )

    future_predictions = (
        latest_actual +
        (future_raw - anchor_prediction)
    )

    if (
        (future_predictions < 0).any()
        or (future_predictions > 110).any()
    ):
        st.warning(
            "Modellen ger extrema värden. "
            "Tolka scenariot mycket försiktigt."
        )

    backtest = walk_forward_backtest(
        country_data,
        selection
    )

except (ValueError, KeyError) as exc:
    st.error(str(exc))
    st.stop()


# ==========================================
# 17. PROGNOSDIAGRAM
# ==========================================

st.subheader(
    "3. Historik och prognos fram till 2036"
)

fig = go.Figure()

# Historiska observationer
fig.add_trace(
    go.Scatter(
        x=observed["Year"],
        y=observed["Life expectancy"],
        name="Historiska värden",
        mode="lines+markers",
        line=dict(
            color=BLUE,
            width=3
        )
    )
)

# Framtidsprognos
fig.add_trace(
    go.Scatter(
        x=[
            latest_actual_year,
            *forecast_years
        ],
        y=[
            latest_actual,
            *future_predictions
        ],
        name="Scenario-prognos",
        mode="lines+markers",
        line=dict(
            color=GREEN,
            width=3,
            dash="dash"
        )
    )
)


# ==========================================
# 18. ILLUSTRATIVT OSÄKERHETSSPANN
# ==========================================

if len(backtest) >= 3:

    rmse = float(
        np.sqrt(
            mean_squared_error(
                backtest["Faktiskt"],
                backtest["Prognos"]
            )
        )
    )

    horizon = (
        forecast_years - latest_actual_year
    )

    widths = rmse * np.sqrt(horizon)

    upper = future_predictions + widths
    lower = future_predictions - widths

    fig.add_trace(
        go.Scatter(
            x=forecast_years,
            y=upper,
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip"
        )
    )

    fig.add_trace(
        go.Scatter(
            x=forecast_years,
            y=lower,
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(56,189,248,0.12)",
            name="Illustrativt osäkerhetsspann",
            hoverinfo="skip"
        )
    )


design_plot(fig)

st.plotly_chart(
    fig,
    use_container_width=True
)

if len(backtest) >= 3:

    st.caption(
        "Det skuggade området är ett illustrativt "
        "osäkerhetsspann baserat på historiska "
        "ettårsfel. Det är inte ett statistiskt "
        "kalibrerat 80 % eller 95 % prediktionsintervall."
    )


# ==========================================
# 19. KPI – PROGNOSRESULTAT
# ==========================================

c1, c2, c3 = st.columns(3)

c1.metric(
    f"Senaste faktiska ({latest_actual_year})",
    f"{latest_actual:.1f} år"
)

c2.metric(
    f"Scenario {end_year}",
    f"{future_predictions[-1]:.1f} år"
)

c3.metric(
    "Skillnad mot senaste faktiska",
    f"{future_predictions[-1] - latest_actual:+.1f} år"
)


# ==========================================
# 20. HISTORISK MODELLVALIDERING
# ==========================================

st.divider()

st.subheader(
    "4. Historisk validering av prognosmodellen"
)

if len(backtest) >= 3:

    mae = mean_absolute_error(
        backtest["Faktiskt"],
        backtest["Prognos"]
    )

    rmse = np.sqrt(
        mean_squared_error(
            backtest["Faktiskt"],
            backtest["Prognos"]
        )
    )

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "MAE – 1 år framåt",
        f"{mae:.2f} år"
    )

    m2.metric(
        "RMSE – 1 år framåt",
        f"{rmse:.2f} år"
    )

    m3.metric(
        "Antal testår",
        len(backtest)
    )

    check = go.Figure()

    check.add_trace(
        go.Scatter(
            x=backtest["År"],
            y=backtest["Faktiskt"],
            mode="lines+markers",
            name="Faktiskt",
            line=dict(color=BLUE)
        )
    )

    check.add_trace(
        go.Scatter(
            x=backtest["År"],
            y=backtest["Prognos"],
            mode="lines+markers",
            name="Historisk prognos",
            line=dict(
                color=RED,
                dash="dash"
            )
        )
    )

    design_plot(check, height=320)

    st.plotly_chart(
        check,
        use_container_width=True
    )

    st.caption(
        "Varje testår prognostiseras med enbart "
        "tidigare års observationer. Ettårsvalidering "
        "garanterar inte att en tolvårsprognos "
        "blir lika träffsäker."
    )

else:

    st.warning(
        "För få historiska valideringsår "
        "för en tillförlitlig feluppskattning."
    )


# ==========================================
# 21. PROGNOSTABELL
# ==========================================

st.divider()

st.subheader(
    "5. Prognosdata"
)

forecast_table = pd.DataFrame({
    "År": forecast_years,
    "Prognostiserad livslängd (år)": np.round(
        future_predictions,
        2
    )
})

for col in selection:

    forecast_table[
        f"Antagande: {DATA_COLUMNS[col]}"
    ] = np.round(
        future_x[col].to_numpy(),
        2
    )

st.dataframe(
    forecast_table,
    hide_index=True,
    use_container_width=True
)


# ==========================================
# 22. EXPORTERA DATA
# ==========================================

st.download_button(
    "Ladda ner prognosen (CSV)",
    data=forecast_table.to_csv(
        index=False
    ).encode("utf-8-sig"),
    file_name="framtidsprognos_2036.csv",
    mime="text/csv"
)


# ==========================================
# 23. METOD OCH BEGRÄNSNINGAR
# ==========================================

with st.expander(
    "Metod och modellbegränsningar"
):

    st.write(
        "**Metod:** Ridge-regression tränad separat "
        "för valt land med årtal och valda "
        "logaritmerade indikatorer."
    )

    st.write(
        "Framtida BNP, vårdutgifter och befolkning "
        "beräknas utifrån användarens antagna "
        "årliga förändringstakter."
    )

    st.write(
        "Prognosen förankras vid den senaste "
        "observerade livslängden."
    )

    st.write(
        "**Begränsningar:** Få historiska observationer, "
        "pandemieffekter, samvariation mellan "
        "förklaringsvariabler och långa prognoshorisonter "
        "gör resultaten osäkra."
    )

    st.write(
        "Modellens samband innebär inte att en "
        "ökning av BNP eller sjukvårdsutgifter "
        "automatiskt orsakar en viss ökning "
        "av livslängden."
    )

    st.write(
        "**Datakällor:** World Bank WDI och "
        "Kaggle/WHO. Din befintliga Random Forest "
        "används inte av denna prognossida."
    )
