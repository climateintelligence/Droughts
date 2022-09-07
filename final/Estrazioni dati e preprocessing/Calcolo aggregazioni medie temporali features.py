import os
import pandas as pd
import numpy as np

# in questo caso calcolo le aggregazioni delle piogge ma basta sostituire il nome del file e delle variabili
os.getcwd()
folders = [x for x in os.listdir('.') if not (x.endswith('.py'))]

for folder in folders:
    # lettura file di testo
    filename = os.listdir(os.path.join(".", folder))[0]
    df = pd.read_csv(os.path.join(os.getcwd(), folder, filename), sep=" ", parse_dates= {"Date" : ["Y","m","dy"]})
    
    # aggiungo colonne iniziali
    df['Date'] = pd.to_datetime(df['Date'])
    df['weekly_mean'] = df.mean(axis=1)
    df = df[['Date','weekly_mean']]
    df = df.resample('W', on='Date').mean().reset_index()
    df['Year'] = pd.DatetimeIndex(df['Date']).year
    df['Week'] = pd.DatetimeIndex(df['Date']).week

    # calcolo medie su diversi lag temporali
    df['Prec1w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-1,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)
    df['Prec4w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-4,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)
    df['Prec8w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-8,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)
    df['Prec12w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-12,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)
    df['Prec16w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-16,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)
    df['Prec24w_' + folder] = df.apply(lambda x: df.loc[(df['Date']>= x.Date+np.timedelta64(-24,'W')) & (df['Date'] < x.Date), 'weekly_mean'].mean(), axis=1)

    # cancello la colonna della media della settimana corrente
    df.drop('weekly_mean', axis=1, inplace=True)

    # salvo su pickle
    varname = filename.split("_")[0] # nome variabile (es. pav=piogge, tas=temp)
    df.to_pickle(varname + "_" + folder + ".pickle")