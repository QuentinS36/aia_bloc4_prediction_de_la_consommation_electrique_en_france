import pandas as pd
from get_hourly_consumption_data import fetch_consumption_data
from get_hourly_weather_data_by_departments import fetch_department_weather

def get_dataset(days: int = 35):
    consumption_data = fetch_consumption_data(days=days)[["start_date", "end_date", "value"]]
    weather_data = fetch_department_weather(days=days)[["Department Code", "Timestamp", "Temperature (°C)"]]

    consumption_data.rename(columns={"Timestamp": "end_date"}, inplace=True)
    weather_data.rename(columns={"Department Code": "dept_code", "Temperature (°C)": "temperature"}, inplace=True)

    consumption_data["start_date"] = pd.to_datetime(consumption_data["start_date"], utc=True).dt.tz_convert(None)
    consumption_data["end_date"] = pd.to_datetime(consumption_data["end_date"], utc=True).dt.tz_convert(None)
    weather_data["Timestamp"] = pd.to_datetime(weather_data["Timestamp"])

    weather_data_grouped_by_timestamp = weather_data.groupby('Timestamp')['temperature'].agg(
        [
            'mean',
            'min',
            'max',
            'median',
            'std',
            ('q1', lambda x: x.quantile(0.25)),
            ('q3', lambda x: x.quantile(0.75))
        ]
    ).reset_index()

    df = pd.merge(consumption_data, weather_data_grouped_by_timestamp, left_on='end_date', right_on="Timestamp", how='left')
    df.drop(columns=['Timestamp'], inplace=True, axis=1)

    return df.reset_index(drop=True)
