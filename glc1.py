import pandas as pd
import numpy as np
import json


def get_data():
    print('get_data')
    df = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv')
    print('read')
    df = df[['event_date', 'event_time','event_title', 'event_description', 'location_description', 'landslide_size', 
        'fatality_count', 'injury_count', 'country_name', 'country_code', 'longitude', 'latitude']]
    df['year'] = df.event_date.apply(lambda x: x[6: 11])
    df['database'] = 'Global Landslide Catalog'
    df['startYear'] = df.event_date.apply(lambda x: x[6: 10])
    df['startMonth'] = df.event_date.apply(lambda x: x[3: 5])
    df['startDay'] = df.event_date.apply(lambda x: x[: 2])
    df.rename(columns={'event_description': 'description', 'location_description': 'location', 'fatality_count': 'casualties'
    , 'injury_count': 'injuredPeopleNumber', 'country_name': 'country'}, inplace=True)
    df.rename(columns={'location_description': 'location'}, inplace=True)
    df['startTime'] = df.event_date.apply(lambda x: x[11: ])
    df['startDate'] = df.event_date.apply(lambda x: x[:10])
    df.drop(columns=['event_date', 'event_time', 'event_title', 'country_code', ], inplace=True)
    df.rename(columns={'landslide_size': 'disaster_Size', 'description': 'event_Description'}, inplace=True)
    df.drop(columns=['year'], inplace=True)
    df.rename(columns={'country_name': 'country'}, inplace=True)
    return df

def return_json():  
    df = get_data()
    to_return = df.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return