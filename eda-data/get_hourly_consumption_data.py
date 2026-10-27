import datetime as dt
import pandas as pd
import requests
import os
from dotenv import load_dotenv
from aggregate_to_hourly import aggregate_to_hourly

def fetch_consumption_data(days: int = 35, margin_days: int = 1):

    load_dotenv()

    url_token_oauth = "https://digital.iservices.rte-france.com/token/oauth/"
    headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + os.getenv("RTE_SECRET_KEY")
        }
    response_token = requests.post(url_token_oauth, headers=headers)
    data = response_token.json()
    token_oauth = data["access_token"]

    end_date = dt.datetime.now(tz=dt.timezone.utc).replace(minute=0, second=0, microsecond=0)
    start_date = end_date - dt.timedelta(days=days)
    start_date_with_margin = end_date - dt.timedelta(days=days + margin_days)
    end_date_str = end_date.isoformat()
    start_date_with_margin_str = start_date_with_margin.isoformat()

    api_url = "https://digital.iservices.rte-france.com/open_api/consumption/v1/short_term"

    headers = {
        "Authorization": "Bearer " + token_oauth
    }

    params = {
        "type": 'REALISED',
        "start_date": start_date_with_margin_str,
        "end_date": end_date_str
    }

    response = requests.get(api_url, headers=headers, params=params)
    data = response.json()

    df = pd.DataFrame({'start_date': [], 'end_date': [], 'updated_date': [], 'value': []})

    for d in data['short_term'][0]['values']:
        the_end_date = dt.datetime.fromisoformat(d['end_date'])
        if start_date < the_end_date < end_date:
            df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)
    df = aggregate_to_hourly(df, agg_func={"updated_date": "max", "value": "mean"})

    return df.reset_index(drop=True)
