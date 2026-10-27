from json import load
from time import sleep
import pandas as pd
import requests
from datetime import datetime, timezone
import zoneinfo
import os
from dotenv import load_dotenv

load_dotenv()

class ExtractorRTE():

    def __init__(self):
        self.generate_oauth_token()
        self.extract_file_name = ''
        self.data = [] # first json response key to access data

    # Generate OAuth Token
    def generate_oauth_token(self):
        url_token = os.getenv("URL_TOKEN_OAUTH")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": "Basic " + os.getenv("RTE_API_KEY")
        }
        try:
            response_token = requests.post(url_token, headers=headers)
            if response_token.status_code == 200:
                data = response_token.json()
                self.oauth_token = data["access_token"]
                print("Token generated !")
            else:
                raise Exception(f"Token Error: {response_token.status_code}")
        except Exception as e:
            raise(e)
    

    # get historical data
    def get_historical_consumption(self, start_date: str, end_date: str, consumption_unit="power"):
        if consumption_unit == 'power':
            api_url = os.getenv("URL_API_POWER_CONSUMPTION")
        elif consumption_unit == 'energy':
            api_url = os.getenv("URL_API_ENERGY_CONSUMPTION")
        else:
            raise Exception(f"Invalid consumption unit: {consumption_unit}")
        #print(f"oAuth TOKEN: {self.oauth_token}")

        headers = {
            "Authorization": "Bearer " + self.oauth_token
        }

        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        print(params)
        try:
            response = requests.get(api_url, headers=headers, params=params)
            if response.status_code == 200:
                print(f"Data extracted from {start_date} to {end_date}!")
                return response.json()
            else:
                data_error = response.json()
                raise Exception(f"API Error {response.status_code}  : {data_error['error']} + ' : ' + {data_error['error_description']}")

        except Exception as e:
            raise(e)

    # get annual data in 2 requests 
    def get_annual_power_consumption(self, year: int):
        data_key = "consolidated_power_consumption"

        # first 6 monhs
        start_date = f"{year-1}-12-31T00:00:00Z"
        end_date = f"{year}-07-01T00:00:00Z"
        data1 = self.get_historical_consumption(start_date, end_date, consumption_unit="power")
        data1_values = data1["consolidated_power_consumption"][0]['values']
        print(f"First half data extracted: {len(data1_values)} records")

        sleep(1)
        
        # second 6 monhs
        start_date = f"{year}-06-30T00:00:00Z"
        end_date = f"{year+1}-01-01T00:00:00Z"
        data2 = self.get_historical_consumption(start_date, end_date, consumption_unit="power")
        data2_values = data2["consolidated_power_consumption"][0]['values']
        print(f"Second half data extracted: {len(data2_values)} records")

        self.extract_file_name = f"{year}_power_consumption"

        self.data = data1_values + data2_values

    # get annual data in 2 requests 
    def get_annual_energy_consumption(self, year: int):
        data_key = "consolidated_energy_consumption"

        start_date = f"{year-1}-12-31T00:00:00Z"
        end_date = f"{year}-07-01T00:00:00Z"
        data1 = self.get_historical_consumption(start_date, end_date, consumption_unit="energy")
        data1_values = data1["consolidated_energy_consumption"][0]['values']
        print(f"First half data extracted: {len(data1_values)} records")

        sleep(1)
        
        # second 6 monhs
        start_date = f"{year}-06-30T00:00:00Z"
        end_date = f"{year+1}-01-01T00:00:00Z"
        data2 = self.get_historical_consumption(start_date, end_date, consumption_unit="energy")
        data2_values = data2["consolidated_energy_consumption"][0]['values']
        print(f"Second half data extracted: {len(data2_values)} records")

        self.extract_file_name = f"{year}_energy_consumption"

        self.data = data1_values + data2_values
        
    # save data to the specific file format
    # sorted data by date
    def save_data_file(self, format='csv'):
        df = pd.DataFrame(self.data)
        df.sort_values('end_date', inplace=True)
        df.reset_index(inplace=True, drop=True)
        if format == 'csv':
            df.to_csv(self.get_data_storage_path('csv'), index=False)
        elif format == 'json':
            df.to_json(self.get_data_storage_path('json'), index=False)
        else:
            raise ValueError(f"Format {format} not supported")

    # build extract file path
    def get_data_storage_path(self, file_extension: str):
        return os.getenv("DATA_STORAGE") + "/" + self.extract_file_name + "." + file_extension

    def format_date(self, date: str, input_format="%Y-%m-%d"):

        dt = datetime.strptime(date, "%Y-%m-%d")
        # Add hour and timezone Paris
        tz_paris = zoneinfo.ZoneInfo("Europe/Paris")
        dt_tz = dt.replace(hour=0, minute=0, second=0, tzinfo=tz_paris)
        return dt_tz.strftime("%Y-%m-%dT%H:%M:%S%z")


try:
    for year in range(2001, 2026):
        Extractor = ExtractorRTE()
        Extractor.get_annual_energy_consumption(year)
        Extractor.save_data_file('csv')
except Exception as e:
    print(e)