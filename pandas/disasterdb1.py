import pandas as pd
import numpy as np
import json

def get_data():
    df = pd.read_excel(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 7\Disaster-database-UP-TO-DATE_1 (2).xlsx')
    df.columns
    df.drop(df.columns[[20 - 16384]], axis = 1, inplace = True)
    df.dropna(how="all", axis=1, inplace = True)
    df['database'] = 'Disaster-database-UP-TO-DATE_1'
    df.columns
    df.rename(columns={'Type': 'disaster_type', 'Internal Displacements': 'internal Displacements' ,'Incident': 'location',
     'Year': 'startyear', 'Injured': 'injuredPeopleNumber' , 'Fatalities': 'casualties', 'Country': 'country', 'StartDate': 'startdate'}, inplace=True)
    df.drop(columns=['City', 'Link 1 General', 'Link 2 General', 'Report', 'Link 3 Image', 'Book Reference 1', 'Library code', 'Book Reference 2', 'Library Code 2'], inplace=True)
    df.drop(columns=['Link 4 Video' ,'Notes'], inplace=True)
    return df
  
def return_json():  
    df = get_data()
    to_return = df.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return