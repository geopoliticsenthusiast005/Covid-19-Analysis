# 📊 COVID-19 Global Data Analysis & Reporting

# 1️⃣ Setup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Optional: Display settings
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")

# 2️⃣ Data Loading & Exploration
# Replace with your dataset path
data_path = "owid-covid-data.csv"
df = pd.read_csv(data_path)

print("Columns:", df.columns)
print("\nSample rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum().head(15))

# 3️⃣ Data Cleaning
df['date'] = pd.to_datetime(df['date'])

# Example: focus on specific countries
countries = ["Kenya", "United States", "India"]
df_countries = df[df['location'].isin(countries)].copy()  # avoid chained assignment warnings

# Handle missing numeric values
num_cols = df_countries.select_dtypes(include=['number']).columns
df_countries[num_cols] = df_countries[num_cols].fillna(0)

# 4️⃣ Exploratory Data Analysis (EDA)
plt.figure(figsize=(12,6))
for c in countries:
    subset = df_countries[df_countries['location'] == c]
    plt.plot(subset['date'], subset['total_cases'], label=c)
plt.title("Total COVID-19 Cases Over Time")
plt.xlabel("Date")
plt.ylabel("Total Cases")
plt.legend()
plt.show()

# Death Rate Comparison
df_countries['death_rate'] = (df_countries['total_deaths'] / df_countries['total_cases']).replace([float('inf')], pd.NA).fillna(0)

plt.figure(figsize=(12,6))
for c in countries:
    subset = df_countries[df_countries['location'] == c]
    plt.plot(subset['date'], subset['death_rate'], label=c)
plt.title("COVID-19 Death Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Death Rate")
plt.legend()
plt.show()

# 5️⃣ Vaccination Progress
plt.figure(figsize=(12,6))
for c in countries:
    subset = df_countries[df_countries['location'] == c]
    plt.plot(subset['date'], subset['total_vaccinations'], label=c)
plt.title("COVID-19 Vaccinations Over Time")
plt.xlabel("Date")
plt.ylabel("Total Vaccinations")
plt.legend()
plt.show()

# 6️⃣ Optional Choropleth Map (latest snapshot)
latest_date = df['date'].max()
latest_df = df[df['date'] == latest_date]

fig = px.choropleth(
    latest_df,
    locations="iso_code",
    color="total_cases",
    hover_name="location",
    title=f"Global COVID-19 Total Cases ({latest_date.date()})",
    color_continuous_scale="Reds"
)
fig.show()

# 7️⃣ Extended Visuals
# 🔹 Top 10 countries by total cases
latest_sorted = latest_df.sort_values(by="total_cases", ascending=False).head(10)
plt.figure(figsize=(10,6))
sns.barplot(data=latest_sorted, x="total_cases", y="location", palette="Reds_r")
plt.title("Top 10 Countries by Total COVID-19 Cases")
plt.xlabel("Total Cases")
plt.ylabel("Country")
plt.show()

# 🔹 Correlation heatmap of key metrics
key_metrics = df_countries[["total_cases", "total_deaths", "total_vaccinations", "new_cases", "new_deaths"]]
plt.figure(figsize=(8,6))
sns.heatmap(key_metrics.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Between Key COVID-19 Metrics")
plt.show()

# 8️⃣ Data Quality Checks
assert (df['total_cases'].fillna(0) >= 0).all(), "Negative total_cases found!"
assert (df['total_deaths'].fillna(0) >= 0).all(), "Negative total_deaths found!"
assert (df['total_vaccinations'].fillna(0) >= 0).all(), "Negative vaccinations found!"
assert df['date'].min() >= pd.to_datetime("2020-01-01"), "Unexpected early dates in dataset!"

print("Notebook ready: Visuals extended and data quality checks added. Add narrative insights in Markdown cells.")

# 9️⃣ Pytest: Write automated data integrity tests to /tests
# Why: lets learners validate data outside the notebook and in CI.
import os
from pathlib import Path

project_root = Path.cwd()
tests_dir = project_root / "tests"
tests_dir.mkdir(exist_ok=True)

covid_csv_env = os.getenv("COVID_CSV", data_path)  # env override for CI

conftest_py = f'''\
import os
import pandas as pd
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def csv_path() -> Path:
    path = Path(os.getenv("COVID_CSV", "{covid_csv_env}"))
    assert path.exists(), f"CSV not found at: {path}. Set COVID_CSV env var or place file accordingly."
    return path

@pytest.fixture(scope="session")
def df(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    return df
'''

(tests_dir / "conftest.py").write_text(conftest_py, encoding="utf-8")

unit_tests_py = '''\
import pandas as pd
import numpy as np

# ---- Basic sanity ----

def test_no_negative_key_metrics(df: pd.DataFrame):
    for col in [
        'total_cases','total_deaths','total_vaccinations',
        'new_cases','new_deaths','new_vaccinations'
    ]:
        if col in df.columns:
            assert (df[col].fillna(0) >= 0).all(), f"Negative values in {col}"


def test_date_range(df: pd.DataFrame):
    assert df['date'].min() >= pd.to_datetime('2020-01-01')
    assert df['date'].max() <= pd.Timestamp.today() + pd.Timedelta(days=1)


def test_unique_keys(df: pd.DataFrame):
    # Why: duplicates across (location,date) can break time-series logic
    subset = df[['location','date']].astype({'location':'string'})
    dup = subset.duplicated().sum()
    assert dup == 0, f"Found {dup} duplicate (location,date) rows"


# ---- Cumulative fields should be non-decreasing per location ----

def _assert_monotonic_non_decreasing(series: pd.Series):
    s = series.dropna().astype(float)
    if s.empty:
        return
    diffs = s.diff().fillna(0)
    assert (diffs >= -1e-9).all(), "Cumulative metric decreased within a location"


def test_cumulative_monotonicity(df: pd.DataFrame):
    for col in ['total_cases','total_deaths','total_vaccinations']:
        if col in df.columns:
            for _, g in df.sort_values('date').groupby('location'):
                _assert_monotonic_non_decreasing(g[col])


# ---- ISO code sanity ----

def test_iso_code_format(df: pd.DataFrame):
    if 'iso_code' not in df.columns:
        return
    iso = df['iso_code'].dropna().astype(str)
    # Allow OWID_* aggregates but require 3-letter codes for countries
    regular = iso[~iso.str.startswith('OWID_')]
    assert regular.str.len().eq(3).all(), "Non-aggregate rows must have 3-letter ISO codes"


# ---- Death rate bounds ----

def test_death_rate_bounds(df: pd.DataFrame):
    if {'total_deaths','total_cases'}.issubset(df.columns):
        with np.errstate(divide='ignore', invalid='ignore'):
            rate = df['total_deaths'] / df['total_cases']
            rate = rate.replace([np.inf, -np.inf], np.nan).fillna(0)
            assert ((rate >= 0) & (rate <= 1)).all(), "Death rate out of [0,1] range"
'''

(tests_dir / "test_data_quality.py").write_text(unit_tests_py, encoding="utf-8")

print(f"Wrote pytest suite to: {tests_dir}
Run: pytest -q")
