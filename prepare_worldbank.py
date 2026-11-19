
# ============================================
# WORLD BANK – FÖRBERED DATA FÖR DASHBOARD
# ============================================

import csv
import re
import zipfile
from pathlib import Path

import pandas as pd


# 1. SÖKVÄGAR

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "life_data"

ZIP_FILE = DATA_DIR / "WDI_CSV.zip"
LOCAL_FILE = DATA_DIR / "life_expectancy_enriched.csv"
OUTPUT_FILE = DATA_DIR / "dashboard_worldbank.csv"


# 2. INDIKATORER

INDICATORS = {
    "SP.DYN.LE00.IN": "Life expectancy",
    "NY.GDP.PCAP.CD": "GDP_Final",
    "SP.POP.TOTL": "Population_Final",
    "SH.XPD.CHEX.PC.CD": "Health_Expenditure",
    "SP.DYN.IMRT.IN": "Infant_Mortality_Rate",
    "SH.IMM.MEAS": "Measles_Vaccination"
}

START_YEAR = 2000
END_YEAR = 2024


# 3. KONTROLLERA FILER

if not ZIP_FILE.exists():
    raise FileNotFoundError(
        f"ZIP-filen hittades inte: {ZIP_FILE}"
    )

if not LOCAL_FILE.exists():
    raise FileNotFoundError(
        f"Den ursprungliga datafilen saknas: {LOCAL_FILE}"
    )


# 4. HÄMTA LÄNDER FRÅN BEFINTLIG DATA

local_df = pd.read_csv(
    LOCAL_FILE,
    usecols=["Country_Code"]
)

country_codes = set(
    local_df["Country_Code"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.upper()
)

print("Antal landskoder i projektet:", len(country_codes))


# 5. ÖPPNA ZIP-FILEN

records = []

with zipfile.ZipFile(ZIP_FILE, "r") as archive:

    csv_files = [
        name for name in archive.namelist()
        if name.lower().endswith("wdicsv.csv")
        or name.lower().endswith("wdidata.csv")
    ]

    if not csv_files:
        raise ValueError(
            "Hittar inte WDICSV.csv i ZIP-filen."
        )

    selected_file = csv_files[0]

    print("Läser:", selected_file)

    with archive.open(selected_file) as raw:

        import io

        text_file = io.TextIOWrapper(
            raw,
            encoding="utf-8-sig",
            errors="replace",
            newline=""
        )

        reader = csv.reader(text_file)

        # Hitta den faktiska rubrikraden
        header = None

        for row in reader:
            if "Indicator Code" in row and "Country Code" in row:
                header = row
                break

        if header is None:
            raise ValueError(
                "Kunde inte hitta kolumnrubrikerna."
            )

        code_index = header.index("Country Code")
        indicator_index = header.index("Indicator Code")

        # WDI kan skriva exempelvis 2024 [YR2024]
        year_columns = {}

        for index, column in enumerate(header):

            match = re.match(
                r"^(\d{4})(?:\s*\[YR\d{4}\])?$",
                column.strip()
            )

            if match:
                year = int(match.group(1))

                if START_YEAR <= year <= END_YEAR:
                    year_columns[index] = year

        print(
            "Årskolumner hittade:",
            len(year_columns)
        )

        if not year_columns:
            raise ValueError(
                "Inga årskolumner hittades."
            )

        # Läs bara de indikatorer som vi behöver
        for row in reader:

            if len(row) <= max(code_index, indicator_index):
                continue

            country_code = row[code_index].strip().upper()
            indicator_code = row[indicator_index].strip()

            if country_code not in country_codes:
                continue

            if indicator_code not in INDICATORS:
                continue

            for index, year in year_columns.items():

                if index >= len(row):
                    continue

                value = row[index].strip()

                if not value:
                    continue

                try:
                    value = float(value)
                except ValueError:
                    continue

                records.append({
                    "Country_Code": country_code,
                    "Year": year,
                    "Indicator": INDICATORS[indicator_code],
                    "Value": value
                })


# 6. SKAPA DASHBOARD-DATA

if not records:
    raise ValueError(
        "Inga användbara värden hittades i filen."
    )

long_df = pd.DataFrame(records)

dashboard_df = long_df.pivot_table(
    index=["Country_Code", "Year"],
    columns="Indicator",
    values="Value",
    aggfunc="first"
).reset_index()

dashboard_df.columns.name = None

dashboard_df = dashboard_df.sort_values(
    ["Country_Code", "Year"]
)

# Säkerställ att alla förväntade kolumner finns
for column in INDICATORS.values():

    if column not in dashboard_df.columns:
        dashboard_df[column] = float("nan")


# 7. SPARA CSV-FILEN

dashboard_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# 8. KONTROLLERA RESULTAT

print("\n===================================")
print("WORLD BANK – KLART!")
print("===================================")

print("Sparad fil:", OUTPUT_FILE)

print("Antal observationer:", len(dashboard_df))

print(
    "Antal länder:",
    dashboard_df["Country_Code"].nunique()
)

print(
    "Tidsperiod:",
    dashboard_df["Year"].min(),
    "–",
    dashboard_df["Year"].max()
)

print("\nAntal värden per indikator:")

for column in INDICATORS.values():

    print(
        f"{column}: "
        f"{dashboard_df[column].notna().sum()}"
    )

print("\nSenaste år med livslängdsdata:")

life_data = dashboard_df.dropna(
    subset=["Life expectancy"]
)

if not life_data.empty:

    print(
        int(life_data["Year"].max())
    )

print("\nFörhandsvisning:")
print(dashboard_df.head())
