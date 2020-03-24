import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.linear_model import LinearRegression
import lightgbm as lgbm


### Compute the metric##################################################################################################
def smape(satellite_predicted_values, satellite_true_values):
    # the division, addition and subtraction are pointwise
    return np.abs((satellite_predicted_values - satellite_true_values)
        / (np.abs(satellite_predicted_values) + np.abs(satellite_true_values)))

def display_smape(prediction,val_label):
    preds = prediction.melt(['id','sat_id'], var_name='target', value_name='y_pred').sort_values(['id','sat_id'])
    true = val_label.melt(['id','sat_id'], var_name='target', value_name='y_true').sort_values(['id','sat_id'])
    values = pd.merge(preds, true, how='inner', on=['id', 'sat_id', 'target'])
    score = 100*(1-values.groupby('sat_id').apply(lambda g: smape(g['y_true'], g['y_pred'])).mean())
    print(f'Score : {score}')
##############################################################################################################################

# Compute some features and calcul
def calcul_by_satellite(df,calcul):
    return df.groupby('sat_id').apply(calcul)

def cycle_calcul(dataIn):
    #dataIn["dt"] = convert_in_min(dataIn["epoch"]-dataIn["epoch"].shift(1))
    #dataIn["period"] = period_24(dataIn["dt"].to_numpy())
    dataIn["cycle"] = cycle(dataIn['period'])
    return dataIn

def period_dt_calcul(dataIn):
    dataIn["dt"] = convert_in_min(dataIn["epoch"]-dataIn["epoch"].shift(1))
    dataIn["period"] = period_24(dataIn["dt"].to_numpy())
    return dataIn


def features(dataIn,local=True):
    # features example
    # dataIn["ro"] = dataIn.apply(lambda row: np.sqrt(row["x"]**2+row["y"]**2+row["z"]**2),axis=1)
    # dataIn["theta"] = dataIn.apply(lambda row: np.arccos(row["z"]/row["ro"]),axis=1)
    # dataIn["phi"] = dataIn.apply(lambda row: np.arctan(row["y"]/row["x"]),axis=1)
    return dataIn

def features_period(dataIn):
    # features example
    #dataIn["x_mean_cycle"] = dataIn["x"].mean()
    #dataIn["y_mean_cycle"] = dataIn["y"].mean()
    #dataIn["z_mean_cycle"] = dataIn["z"].mean()
    #dataIn["Vx_mean_cycle"] = dataIn["Vx"].mean()
    #dataIn["Vy_mean_cycle"] = dataIn["Vy"].mean()
    #dataIn["Vz_mean_cycle"] = dataIn["Vz"].mean()

    return dataIn

### Compute feature for each satellite
def feature_by_satellite(df):
    return df.groupby('sat_id').apply(features)

def features_by_period(df):

    return df.groupby(['period','sat_id']).apply(features_period)

############################################################################################################
#Time and cycle period etc
def convert_in_min(s):
    return np.floor(s.dt.total_seconds()/60)

## time separation of data
def separate(train,days):
    train["day"] = train["epoch"].apply(lambda x: x.day)
    train1 = pd.DataFrame()
    train2 = pd.DataFrame()
    train1 = train.query(f'day <={days} ')
    train2 = train.query(f'day >{days}')
    train1 = train1.drop(["day","epoch"],axis=1)
    train2 = train2.drop(["day","epoch"],axis=1)
    del train["day"]
    return train1,train2


### Compute the period
def period_24(x):
    res = [0]
    n = len(x)
    k = 1
    for i in range (1,n):
        if x[i]< 5:
            res.append(res[-1])
        else:
            res.append(k%24)
            k+=1
    return res
### Compute cycle
def cycle(s):
    res = []
    num = 0
    previous = 0
    for p in s:
        if p==0 and previous !=0:
            num+=1
        res.append(num)
        previous = p
    return res



#### Period extraction
def get_first_period(subtrain):
    i=0
    n = 0
    while subtrain.iloc[0]['period'] != subtrain.iloc[24+i]['period']:
        i+=1
    return subtrain.iloc[0:24+i]


def extract_first_period(subtrain,subtest):
    to_copy = get_first_period(subtrain)
    for per in to_copy["period"][::-1].unique():
        subtest.loc[subtest['period'] ==per,["x","y","z","Vx","Vy","Vz"]] = to_copy.loc[to_copy['period']==per,["x","y","z","Vx","Vy","Vz"]].values[-1]
    return subtest

def repeat_first_per(train,test):
    for sat in tqdm_notebook(test["sat_id"].unique()):
        tmp = train.query('sat_id == @sat')
        test.loc[test["sat_id"] == sat] = extract_first_period(tmp,test.loc[test["sat_id"] == sat])
    return test

def get_last_period(subtrain):
    i=0
    n = subtrain.shape[0]
    while subtrain.iloc[n-1]['period'] != subtrain.iloc[n-24-i]['period']:
        i+=1
    return subtrain.iloc[n-23-i:n]


def extract_last_period(subtrain,subtest):
    to_copy = get_last_period(subtrain)
    for per in to_copy["period"].unique():
        subtest.loc[subtest['period'] ==per,["x","y","z","Vx","Vy","Vz"]] = to_copy.loc[to_copy['period']==per,["x","y","z","Vx","Vy","Vz"]].values[-1]
    return subtest

def repeat_last_per(train,test):
    for sat in tqdm_notebook(test["sat_id"].unique()):
        tmp = train.query('sat_id == @sat')
        test.loc[test["sat_id"] == sat] = extract_last_period(tmp,test.loc[test["sat_id"] == sat])


