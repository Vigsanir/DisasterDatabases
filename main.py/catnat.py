import pandas as pd
import numpy as np
import json

def get_data():
    df = pd.read_excel(r'C:\Users\vigsa\Desktop\diplwmatikh arxeia transform\bash 3-gallika\BD Catnat_exemple (2).xlsx')
    df['database'] = 'BD Catnat_exemple'
   
    df.drop(columns=['Danger', 'nber_pers_assigned', 'human_csq_index', 'indice_csq_materials', 'index_evt',], inplace=True)
    df.rename(columns={'number_injuries': 'injuredPeopleNumber'}, inplace=True)
    df.rename(columns={'number_victims': 'casualties'}, inplace=True)
    df.rename(columns={'year': 'startyear'}, inplace=True)
    df.rename(columns={'month': 'startmonth'}, inplace=True)
    df.drop(columns=['number_of_homeless', 'under_peril'], inplace=True)
    df['Endyear'] = df["end date"].apply(lambda x: x[:4])
    df['EndDay'] = df["end date"].apply(lambda x: x[8:])
    df['EndMonth'] = df["end date"].apply(lambda x: x[5:7])
    df['StartYear'] = df["Start date"].apply(lambda x: x[:4])
    df['StartDay'] = df["Start date"].apply(lambda x: x[8:])
    df['StartMonth'] = df["Start date"].apply(lambda x: x[5:7])
    df.rename(columns={'hazard': 'disaster_Type'}, inplace=True)
    df.rename(columns={'Disaster_type': 'disaster_Type'}, inplace=True)
    df.drop(columns=['number_evacuated'], inplace=True)
    return df    
    
def return_json():
    df = get_data()
    to_return = df.to_json(orient='records')
    to_return = json.loads(to_return)
    return to_return
