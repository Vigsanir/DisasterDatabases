import pandas as pd
import numpy as np
import json


def get_data():
    df_glc = pd.read_csv(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 8\Global_Landslide_Catalog_Export (1).csv')
    df_glc = df_glc[['event_date', 'event_time','event_title', 'event_description', 'location_description', 'landslide_size', 
        'fatality_count', 'injury_count', 'country_name', 'country_code', 'longitude', 'latitude']]
    df_glc['year'] = df_glc.event_date.apply(lambda x: x[6: 11])
    df_glc['database'] = 'Global Landslide Catalog'
    df_glc['startYear'] = df_glc.event_date.apply(lambda x: x[6: 10])
    df_glc['startMonth'] = df_glc.event_date.apply(lambda x: x[3: 5])
    df_glc['startDay'] = df_glc.event_date.apply(lambda x: x[: 2])
    df_glc.rename(columns={'event_description': 'description', 'location_description': 'location', 'fatality_count': 'casualties'
    , 'injury_count': 'injuredPeopleNumber', 'country_name': 'country'}, inplace=True)
    df_glc.rename(columns={'location_description': 'location'}, inplace=True)
    df_glc['startTime'] = df_glc.event_date.apply(lambda x: x[11: ])
    df_glc['startDate'] = df_glc.event_date.apply(lambda x: x[:10])
    df_glc.drop(columns=['event_date', 'event_time', 'event_title', 'country_code', ], inplace=True)
    df_glc.rename(columns={'landslide_size': 'disaster_Size', 'description': 'event_Description'}, inplace=True)
    df_glc.drop(columns=['year'], inplace=True)
    df_glc.rename(columns={'country_name': 'country'}, inplace=True)
    return df_glc

def return_json():  
    df_glc = get_data()
    to_return = df_glc.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return