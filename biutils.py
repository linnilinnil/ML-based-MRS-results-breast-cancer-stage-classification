# update data table, only call this to update instead of storing. 
import sqlite3
import numpy as np
from matplotlib import pyplot as plt 
import pandas as pd 
import seaborn as sns
import numpy as np
import pingouin as pg
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
import random
from sklearn.model_selection import StratifiedShuffleSplit

def data_const(dat):
    # Data consitutes
    #voxels
    malignantv = dat[dat["label"]=="malignant"]["voxel"] # label is name of the column where label is stored
    benignv = dat[dat["label"]=="benign"]["voxel"]
    malignantd = list(malignantv.apply(lambda x:x.split("-")[0]).unique())
    benignd = list(benignv.apply(lambda x:x.split("-")[0]).unique())
    #dataset
    vl = ["Malignant" for m in malignantv] + ["Benign" for b in benignv] 
    dl = ["Malignant" for m in malignantd] + ["Benign" for b in benignd] 
    datasets = pd.DataFrame({"datasets":malignantd+benignd,"labels":dl})
    voxels = pd.DataFrame({"voxels":list(malignantv)+list(benignv),"labels":vl})
    #count how many different kinds there are
    categories = voxels["labels"].value_counts().index
    counts = voxels['labels'].value_counts().values
    #plotting
    fig,ax = plt.subplots(1,2)
    ax[0].bar(categories, counts, width=0.5)
    ax[0].set_title("Total #voxels of different labels")
    categories = datasets["labels"].value_counts().index
    counts = datasets['labels'].value_counts().values
    ax[1].bar(categories, counts, width=0.5)
    ax[1].set_title("Total #datasets of different labels")
    print('number of malignant voxels:')
    print(len(malignantv))
    print('number of benign voxels:')
    print(len(benignv))
    plt.tight_layout()

    return malignantd, benignd

def update_tbl(new_df,tbl_name,drop=False):
    conn = sqlite3.connect("quant.db")
    cur = conn.cursor()
    if drop:
        cur.execute(str("DROP TABLE IF EXISTS "+tbl_name))
        conn.commit()
    new_df.to_sql(tbl_name,conn,index=False)
    #print(pd.read_sql_query("SELECT * FROM no_out WHERE label == 'benign' LIMIT 10",conn))
    conn.close()

def all_tbl():
    conn = sqlite3.connect("quant.db")
    cur = conn.cursor()
    x = cur.execute("SELECT name FROM sqlite_master where type='table'")
    for y in x.fetchall():
        print(y)
    conn.close()

def get_tbl(tbl_name):
    conn = sqlite3.connect("quant.db")
    tbl = pd.read_sql_query(str("Select * from "+tbl_name),conn)
    conn.close()
    return tbl

def group_lenv(data,mlist):
    lenv = pd.DataFrame()
    for met in mlist: 
        dat = data.loc[:,[met,'label']]
        # Levene's Test in Python using Pingouin
        res = pg.homoscedasticity(dat,group='label',dv=met)
        lenv = pd.concat([lenv,res])
    lenv.index = mlist
    #print(lenv[lenv["equal_var"]==True])
    return lenv

def map_adc(data):
    # this is a separate sheet that I created. all data tables in the ds have age+adc mapped to them
    agepath = "/Users/linlin/Downloads/199data/Breast_recon_details_maybe_useful.xlsx"
    additional = pd.read_excel(agepath,sheet_name='Sheet4',index_col=False,names=["ds","ADC","age"],header=None)
    additional['ds'] = additional['ds'].apply(lambda x:"data"+x.split("-")[2])
    #age_dict = dict(zip(list(additional["ds"]),list(additional["age"])))
    adc_dict = dict(zip(list(additional["ds"]),list(additional["ADC"])))
    #data["age"] = data["voxel"].apply(lambda x:x.split("-")[0]).map(age_dict)
    data["adc"] = data["voxel"].apply(lambda x:x.split("-")[0]).map(adc_dict)
    return data

#doing the pipeline with vars from the mwu tests
#This code is just run to get train and test set index
def tts(dat,vars, train_size = 0.8,rs=np.random.RandomState()):
    X = dat[vars]
    y = LabelEncoder().fit_transform(dat.label)
    groups = dat[['dataset']]
    group_dict = dict(zip(dat.dataset,dat.label))
    gps = list(group_dict.keys())
    gl = list(group_dict.values())

    sss = StratifiedShuffleSplit(n_splits=1,train_size=train_size,random_state=rs)
    for i, (train_index, test_index) in enumerate(sss.split(gps, gl)):
        #print(i)
        trds = [gps[idx] for idx in train_index] 
        teds = [gps[idx] for idx in test_index]
        g_train = groups[groups.dataset.isin(trds)].dataset
        g_test = groups[groups.dataset.isin(teds)].dataset
        #print(set(g_train).intersection(set(g_test)))
        X_train = X.loc[dat.dataset.isin(g_train),vars]
        X_test = X.loc[dat.dataset.isin(g_test),vars]
        y_train = y[dat.dataset.isin(g_train)]
        y_test = y[dat.dataset.isin(g_test)]
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train,X_test,y_train,y_test,g_train,g_test



