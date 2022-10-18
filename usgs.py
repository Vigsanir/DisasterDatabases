import pandas as pd
import numpy as np
import json


def get_data():
    df = pd.read_csv(f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime=2004-01-01&endtime=2004-12-31&minmagnitude=6')
    for year in range(2021,2023): 
        df = df.append(pd.read_csv(f'https://earthquake.usgs.gov/fdsnws/event/1/query?format=csv&starttime={year}-01-01&endtime={year}-12-31&minmagnitude=6'),ignore_index = True)
    df['database'] = 'earthquake.usgs.gov'
    df['startYear'] = df.time.apply(lambda x: x[: 4])
    df['startMonth'] = df.time.apply(lambda x: x[5: 7])
    df.drop(columns=['status', 'depthError', 'magError', 'magNst', 'horizontalError', 'magType', 'updated', 'locationSource', 'magSource', 'depth',
    'net', 'dmin', 'rms', 'gap', 'id', 'nst' ], inplace=True)
    df.rename(columns={'place': 'location', 'type':'disaster_type'}, inplace=True)
    return df

def return_json():
    df = get_data()
    to_return = df.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return