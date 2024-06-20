from math import sqrt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from astral import LocationInfo
from astral.sun import sun


def calculate_smape(actual, predicted) -> float: 
  
    # Convert actual and predicted to numpy 
    # array data type if not already 
    if not all([isinstance(actual, np.ndarray),  
                isinstance(predicted, np.ndarray)]): 
        actual, predicted = np.array(actual), np.array(predicted) 
  
    return round(np.mean( np.abs(predicted - actual) / ((np.abs(predicted) + np.abs(actual))/2) )*100, 2)
    

def calculate_rmse(actual, predicted) -> float:
    score = sqrt(mean_squared_error(actual, predicted))
    return score

def calculate_norm_rmse(actual, predicted, min_max=False) -> float:
    score = calculate_rmse(actual, predicted)
    norm_score = score / (actual.max() - actual.min())
    return norm_score
    
def calculate_mase(actual, predicted):
    values = []
    return round(np.mean([
        np.abs(actual[i] - predicted[i]) / (np.abs(actual[i] - actual[i - 1]) / len(actual) - 1)
        for i in range(1, len(actual))
    ]), 2)

def cal_metrics(actual, predictions):
    perf_metrics = pd.DataFrame({
        "MAE": [mean_absolute_error(actual, predictions)],
        "MAPE": [round(mean_absolute_percentage_error(actual, predictions) * 100, 2)],
        "R2_Score": [r2_score(actual, predictions)],
        "SMAPE": [calculate_smape(actual, predictions)],
        "nRMSE": [calculate_norm_rmse(actual, predictions)],
        "RMSE": [calculate_rmse(actual, predictions)],
        "MASE": [calculate_mase(actual, predictions)]
    })
    return perf_metrics


def create_std_scaler(data, column_name):
    data_scaler = StandardScaler()
    if isinstance(column_name, str):
        column_name = [column_name]
    data_scaler = data_scaler.fit(data[column_name])
    transformed_data = data.copy()
    transformed_data[column_name] = data_scaler.transform(data[column_name])
    scaling_info = pd.DataFrame({
        "columns": column_name,
        "mean": data_scaler.mean_,
        "variance": data_scaler.var_,
        "scale_factor": data_scaler.scale_
    })
    print(scaling_info)
    return data_scaler, transformed_data[column_name]


def feature_engineer(data, data_path):
    data["bld_engcons"] = data["bld_engcons"] - data["comms_and_services"]
    data = data.drop(columns=["comms_and_services"])
    circuit_columns = ["space_heating", "hot_water", "sockets", "lighting", "car_chargers", "bld_engcons"]

    # identify the columns with missing data in them
    missing_columns = pd.DataFrame(data.isna().sum().reset_index())
    missing_columns = missing_columns[missing_columns[0] > 0]
    missing_columns = missing_columns["index"].tolist()
    
    # impute the data using linear interpolation method
    for column in missing_columns:
        data[column] = data[column].interpolate(method="linear", limit_direction="both")

    # missing imputation with mode for categorical columns
    for col in ["forecast_visibility", "forecast_winddirection"]:
        data[col] = data[col].fillna(data[col].mode().item())
       
    # drop the forecastperiod column
    data = data.drop(columns=["forecastperiod", "forecast_interval", "forecast_datadate"], axis=0)
    
    # transform the forecast visibility
    # Reference
    # https://www.metoffice.gov.uk/weather/guides/what-does-this-forecast-mean#:~:text=Visibility%20measures%20the%20distance%20at,Less%20than%201%2C000%20metres
    visibility_map = {
        "EX": 6, "VG": 5, "GO": 4, "MO": 3, "PO": 2, "VP": 1
    }
    
    data["forecast_visibility"] = data["forecast_visibility"].map(visibility_map)
    
    # forecast winddirection convertion to degrees 
    degrees_map = pd.read_csv(data_path + "degrees.csv", sep=",")
    degrees_lookup = {abbrv: deg for abbrv, deg in zip(degrees_map["Abbrv."], degrees_map["Degrees"])}
    data["forecast_winddirection"] = data["forecast_winddirection"].map(degrees_lookup)

    # boolean transformation
    data["weekend"] = data["weekend"].astype(int)
    data["bank_holiday"] = data["bank_holiday"].astype(int)

    
    # week of year
    data["week_of_year"] = data.index.isocalendar().week

    # sunlight features
    location = LocationInfo(
        "Swansea",
        "United Kingdom",
        latitude=51.6066,  # Latitude of Swansea
        longitude=-3.9689,   # Longitude of Swansea
        timezone="Europe/London"  # Timezone for Swansea (same as London)
    )
    data["sunrise_hour"] = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunrise'].hour
        for date in data.index
    ]

    data["sunset_hour"] = [
        sun(location.observer, date=date, tzinfo=location.timezone)['sunrise'].hour
        for date in data.index
    ]

    data["daylight_hours"] = data["sunset_hour"] - data["sunrise_hour"]
    data['is_daylight'] = np.where((data.index.hour >= data['sunrise_hour']) & 
      (data.index.hour < data['sunset_hour']), 1, 0)

    # cyclic transformations
    denom_map = {
        "hour": 24, "month": 12, "day_of_week": 7, "forecast_winddirection": 360, "sunrise_hour": 24, "sunset_hour": 24
    }
    for col_name in denom_map:
        data[f"sine_{col_name}"] = np.sin(2 * np.pi * data[col_name] / denom_map[col_name])
        data[f"cos_{col_name}"] = np.cos(2 * np.pi * data[col_name] / denom_map[col_name])

    # drop the raw columns used for encoded columns
    data = data.drop(columns=list(denom_map.keys()))
    return data


def first_preprocess(data, data_path):
    # rename the columns to make it easier for accessing
    data = data.rename(columns={i: i.lower().replace(" ", "_") for i in data.columns})
    
    # set the timestamp as the index for easy access of the data
    data['time'] = pd.to_datetime(data['time'], format='mixed')
    data = data.rename(columns={"time": "datetime"})
    data = data.set_index("datetime")
    data.info()

    data = data[~data.index.duplicated(keep='first')]
    data = data.asfreq("h")
    
    # subset the dataset to load only the post covid dataset
    data = data[data.index >= "2021-01-01"]
    
    # feature engineer the data
    data = feature_engineer(data, data_path)    
    return data