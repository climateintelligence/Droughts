from tqdm import tqdm
import os
import pandas as pd
import xarray as xr

import warnings
warnings.filterwarnings("ignore")

# INPUT : i dataset di NDVI su 16 settimane estratti per ciascuna area di interesse con gli shapefile.
## N.B. : i dataset di NDVI sono disponibili originariamente in netCDF 
# OUTPUT : il dataset unico contenente il segnale di NDVI e l'anomalia di NDVI settimanali su tutte le aree 

files = [f for f in os.listdir() if f.endswith('.nc')]

for f in tqdm(files):
    df = xr.open_dataset(f) # apro il netCDF
    datetimeindex = df.indexes['time'].to_datetimeindex() # sfrutto xarray per calcolare più facilmente la data in un format friendly per pandas
    df = df.to_dataframe() # converto netCDF a datafraeme pandas
    df = df[~df._250m_16_days_NDVI.isna()].reset_index() # scarto quei punti per cui il dato di NDVI non è disponibile
    df = df[['time','_250m_16_days_NDVI']]
    df = df.groupby('time').apply(lambda x: x.mean()).reset_index()
    df['time']=datetimeindex
    df['time'] = pd.to_datetime(df['time']) # mi assicuro che la colonna time sia del tipo DateTime
    df = df.resample('W',on='time').mean().interpolate().reset_index()
    df.rename(columns={'_250m_16_days_NDVI': 'NDVI_' + f.split(".")[0], 'time' : 'Date'},inplace=True) 
    df['Year'] = df['Date'].dt.year
    df['Week'] = df['Date'].dt.week
    mediapluriennale = df.groupby('Week').apply(lambda x: x['NDVI_'+f.split(".")[0]].mean()).reset_index()
    mediapluriennale.rename(columns={0:'media'},inplace=True)
    df['anomalia_'+f.split(".")[0]] = df.apply(lambda x: x['NDVI_'+f.split(".")[0]] - mediapluriennale['media'].loc[mediapluriennale['Week'] == x['Week']].iloc[0], axis=1)
    df.to_csv(f.split(".")[0]+".csv", index=False)