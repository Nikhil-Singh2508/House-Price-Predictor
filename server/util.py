import json
import pickle
import numpy as np
import pandas as pd
__locations = None
__data_columns = None
__model = None


def get_location_names():
    return __locations


def load_saved_artifacts():
    print("load artifacts....")
    global __data_columns
    global __locations

    with open("./artifacts/estate_columns.json", 'r') as f:
        __data_columns = json.load(f)['data_column']
        __locations = __data_columns[3:]

    with open('./artifacts/estate_price_predictor_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    print("loaded...")


def get_price(location, sqft, bhk, bath):
    input_data = pd.DataFrame(columns=__data_columns)

    # Populate the DataFrame with input values
    input_data.loc[0] = [sqft, bath, bhk] + [0] * (len(input_data.columns) - 3)

    # Set the location column to 1
    if location in input_data.columns:
        input_data[location] = 1

    # Fill any missing columns with 0
    input_data = input_data.fillna(0)
    return __model.predict(input_data)[0]


if __name__ == '__main__':
    load_saved_artifacts()
    print(get_location_names())
