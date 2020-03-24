#########Some function we need to process the data#################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
import joblib
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings("ignore")

def calcul_by_satellite(df,calcul):
    return df.groupby('sat_id').apply(calcul)

def cycle_calcul(dataIn):
    dataIn["cycle"] = cycle(dataIn['period'])
    return dataIn

def period_dt_calcul(dataIn):
    dataIn["dt"] = convert_in_min(dataIn["epoch"]-dataIn["epoch"].shift(1))
    dataIn["period"] = period_24(dataIn["dt"].to_numpy())
    return dataIn

def convert_in_min(s):
    return np.floor(s.dt.total_seconds()/60)

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




def get_last_period(subtrain):

    last_period = subtrain['period'].iloc[-1]
    penultimate = subtrain.index[subtrain['period'] == last_period][-2] + 1
    return subtrain.loc[penultimate:]




def repeat_last_per(train, test):
    for sat in test["sat_id"].unique():
        test_sat = test[test["sat_id"] == sat]
        tmp_sat_train = train[train["sat_id"] == sat]
        last_period = get_last_period(tmp_sat_train)
        len_test_sat = len(test_sat)

        new_len = round(len_test_sat / len(last_period) + 0.5)

        repeated_period = pd.concat([last_period] * new_len).reset_index(drop = True).iloc[:len_test_sat]

        repeated_period.set_index(test_sat.index, inplace = True)
        test[test["sat_id"] == sat] = test_sat[['sat_id', 'id','period']].merge(repeated_period[["x","y","z","Vx","Vy","Vz"]], how = "outer", left_index = True, right_index = True)

###################### Main task##############################"

train = pd.read_csv('train.csv')
test = pd.read_csv("test.csv")

train["epoch"] = pd.to_datetime(train["epoch"])
test["epoch"] = pd.to_datetime(test["epoch"])

train.sort_values(['sat_id', 'epoch'],inplace=True)
test.sort_values(['sat_id', 'epoch'],inplace=True)

train["test"] = 0
test["test"] = 1
tmp = pd.concat([train,test],ignore_index=True)


tmp = calcul_by_satellite(tmp,period_dt_calcul)
tmp.drop('dt',axis=1,inplace=True)

train = tmp.query('test == 0')
test =  tmp.query('test == 1')

test.drop(['test'],inplace=True,axis=1)

train.fillna(method='bfill',inplace=True)

repeat_last_per(train,test)
val = test
val.drop(["x_sim","y_sim","z_sim","Vx_sim","Vy_sim","Vz_sim"],axis=1,inplace=True)
del train

val = calcul_by_satellite(test,cycle_calcul)
prediction= test[["id","sat_id"]]

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)

dico = {"x" : ["x","Vx","cycle"],
        "y" : ["y","Vy","cycle"],
        "z" : ["z","Vz","cycle"],
        "Vx" : ["Vx","x","cycle"],
        "Vy" : ["Vy","y","cycle"],
        "Vz" : ["Vz","z","cycle"]}

for sat in test["sat_id"].unique():
    for target in ["x","y","z","Vx","Vy","Vz"]:

        val_sat = val.loc[val['sat_id'] == sat]
        val_sat = val_sat[dico[target]]
        val_sat = poly.fit_transform(val_sat)
        model = joblib.load(f'Models/{sat}-{target}')
        prediction.loc[prediction['sat_id'] == sat,target] = model.predict(val_sat)

prediction.drop('sat_id',inplace=True,axis=1)
prediction.to_csv("submission.csv",index=False)
