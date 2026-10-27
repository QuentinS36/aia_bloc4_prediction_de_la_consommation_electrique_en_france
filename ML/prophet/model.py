import pandas as pd
import plotly.express as px
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# conversion pour prophet
def convert_df_to_prophet(df):
    df_prophet = df.copy()
    df_prophet.drop('start_date', axis=1, inplace=True)
    df_prophet.rename(columns={'end_date': 'ds', 'value': 'y',}, inplace=True)
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], utc=True).dt.tz_convert(None)
    return df_prophet

# Création de train_set et test_set
def create_train_test(df, train_period="30 days", test_period="1 day"):
    """
    Crée une liste de train_set et test_set en partant de la fin du dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Dataset avec une colonne 'ds' (datetime)
    train_period : str
        Durée du train_set (ex: "365 days", "2 years")
    test_period : str
        Durée du test_set (ex: "30 days", "1 month")
    
    Returns:
    --------
    tuple (train_set, test_set)
    """
    
    # Convertir les périodes en timedelta
    train_delta = pd.Timedelta(train_period)
    test_delta = pd.Timedelta(test_period)
    
    df = df.copy()
    
    # Date de fin du dataset
    end_date = df['ds'].max()
    
    # Dates du test_set
    test_start = end_date - test_delta
    test_end = end_date
        
    # Dates du train_set (avant le test_set)
    train_start = test_start - train_delta
    train_end = test_start
        
    # Filtrer les données
    train_set = df[(df['ds'] >= train_start) & (df['ds'] < train_end)].copy()
    test_set = df[(df['ds'] >= test_start) & (df['ds'] < test_end)].copy()
        
    # Ajouter seulement si les sets ne sont pas vides
    if len(train_set) > 0 and len(test_set) > 0:
        return (train_set, test_set)
    else:
        raise Exception("Pas assez de données pour créer les sets")

# train and predict 
def train_predict(train_test_set, regressors=[], hyperparams={}, verbose=True):

    train_set = train_test_set[0]
    test_set = train_test_set[1]

    # model
    model = Prophet(**hyperparams)
    if len(regressors) > 0:
        for regressor in regressors:
            model.add_regressor(regressor)
    model.fit(train_set)
    fields = ['ds'] + regressors
    train_pred = model.predict(train_set[fields])
    test_pred = model.predict(test_set[fields])

    # Calculate metrics
    train_mae = mean_absolute_error(train_set['y'], train_pred['yhat'])
    test_mae = mean_absolute_error(test_set['y'], test_pred['yhat'])
    train_mape = mean_absolute_percentage_error(train_set['y'], train_pred['yhat'])
    test_mape = mean_absolute_percentage_error(test_set['y'], test_pred['yhat'])

    if verbose:
        print(f'Train MAE: {train_mae:.2f}, Test MAE: {test_mae:.2f}')
        print(f'Train MAPE: {100*train_mape:.2f}%, Test MAPE: {100*test_mape:.2f}%')
    else:
        # log
        pass

    df_pred = test_pred[['ds', 'yhat']]
    df_pred.to_csv(os.getenv('DATA_STORAGE') + 'prophet_predictions.csv', index=False)
    return df_pred