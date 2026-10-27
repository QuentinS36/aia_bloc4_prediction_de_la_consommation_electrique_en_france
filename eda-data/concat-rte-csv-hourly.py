import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

data_storage = os.getenv("DATA_STORAGE")
df_final = None

# Concatenate all years
for year in range(2000, 2026):
    df = pd.read_csv(f'{data_storage}/{str(year)}_power_consumption.csv')
    if df_final is None:
        df_final = df
    else:
        df_final = pd.concat([df_final, df])
        
df_final.reset_index(inplace=True, drop=True)

# agregation by hour
df_final['start_hour'] = df_final['start_date'].str.replace(':30:', ':00:')

df_final['start_date'] = pd.to_datetime(df_final['start_date'], utc=True).dt.tz_convert('Europe/Paris')
df_final['end_date'] = pd.to_datetime(df_final['end_date'], utc=True).dt.tz_convert('Europe/Paris')
df_final['start_hour'] = pd.to_datetime(df_final['start_hour'], utc=True).dt.tz_convert('Europe/Paris')

df_final['end_hour'] = df_final['start_hour'] + pd.Timedelta(hours=1)

df_final_hourly = df_final.groupby('start_hour').agg({
    'start_date': 'first',
    'end_date': 'last',
    'value': 'mean',
}).reset_index(drop=True)

df_final_hourly.to_csv(data_storage + '/power_consumption_2000_2025_hourly.csv', index=False)

print(df_final_hourly.shape)