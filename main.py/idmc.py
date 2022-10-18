import pandas as pd
import numpy as np
import json

def get_data():
    df = pd.read_excel(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 6\IDMC_Internal_Displacement_Disasters_Events_2008_2021.xlsx')
    df = df.iloc[1: , :]
    df.rename(columns={'Start Date': 'StartDate', 'Event Name': 'event_description', 'Hazard Type': 'disaster_Type', 'Name': 'country' , 'Hazard Category': 'disaster_category'}, inplace=True)
    df['database'] = 'IDMC_Internal_Displacement'
    df['startYear'] = df.StartDate.apply(lambda x: x[: 4])
    df['startMonth'] = df.StartDate.apply(lambda x: x[5: 7])
    df['startDay'] = df.StartDate.apply(lambda x: x[8:])
    df.drop(columns=['Year'], inplace=True)
    df.drop(columns=['ISO3'], inplace=True)
    return df

def return_json():
    df = get_data()
    to_return = df.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return
