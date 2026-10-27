import pandas as pd

df_om = pd.read_csv("./csv/open-meteo/weather_historical_2000_2026_top50_20260317_121955.csv")
# keep only temperature and save file
df_reducted = df_om[['City', 'Timestamp', 'Temperature (°C)']]
df_reducted.rename(columns={'Temperature (°C)': 'Temperature'}, inplace=True)
df_reducted.to_csv("./csv/open-meteo/weather_historical_2000_2026_top50_reducted.csv", index=False)

# conversion timestamp to datetime with timezone
df_reducted['datetime_tz'] = pd.to_datetime(df_reducted['Timestamp'], utc=True).dt.tz_convert('Europe/Paris')

# aggregation !!!
df_groupby_temp = df_reducted.groupby('datetime_tz')['Temperature'].agg([
    'mean','min','max','median','std',
    ('q1', lambda x: x.quantile(0.25)),
    ('q3', lambda x: x.quantile(0.75))
]).reset_index()

df_groupby_temp.to_csv("./csv/open-meteo/temperature-metrics-2020-2025.csv", index=False)