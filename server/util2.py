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
    print("Loading artifacts...")
    global __data_columns
    global __locations
    global __model

    with open("./artifacts/estate_columns.json", 'r') as f:
        __data_columns = json.load(f)['data_column']
        # Assuming the first 3 columns are non-location features
        __locations = __data_columns[3:]

    with open('./artifacts/estate_price_predictor_model.pickle', 'rb') as f:
        __model = pickle.load(f)

    print("Artifacts loaded successfully.")


def get_price(location, sqft, bhk, bath):
    input_data = pd.DataFrame(columns=__data_columns)

    # Initialize DataFrame with zeros and set input values
    input_data.loc[0] = [0] * len(__data_columns)
    input_data.at[0, 'total_sqft'] = sqft
    input_data.at[0, 'bhk'] = bhk
    input_data.at[0, 'bath'] = bath

    # Ensure the location name matches the format used during training
    # location = location.strip().lower()
    if location in __locations:
        loc_index = __data_columns.index(location)
        input_data.iat[0, loc_index] = 1

    # Fill any missing columns with 0 (although this should not be necessary here)
    input_data = input_data.fillna(0)

    return round(__model.predict(input_data)[0], 2)


if __name__ == '__main__':
    load_saved_artifacts()
    # print(get_location_names())
