#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
00-first-tour.py
"""


"""
find here a first study of the dataset, in which we seek to understand and
give meaning to the dataset.

we are not trying to solve our problem but will focus on visualization,
clenaning and feature engineering.

at first we will just study the corelations, the links, the quality and the
meaning of our dataset.

external research and more general considerations may be included in this work

"""


# import

import os, sys, logging, random, time

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


# logging 

l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info


# graph

# %matplotlib
# sns.set()


# consts

FOLDER      = "Driven-Data-Blood-Donations"
TRAIN_FILE  = "training_data.csv"
TEST_FILE   = "test_data.csv"


# functions

def timer(funct) : 

    def wrapper(*args, **kwargs) : 
        
        t = time.time()

        res = funct(*args, **kwargs)

        t = round(time.time() - t, 2)
        msg = funct.__name__ + " : " + str(t) + " secs" 
        print(msg)

        return res

    return wrapper


def caller(funct) : 

    def wrapper(*args, **kwargs) : 

        msg = funct.__name__ + " : called"
        print(msg)
        

        res = funct(*args, **kwargs)

        msg = funct.__name__ + " : ended"
        print(msg)

        return res

    return wrapper


# @timer
def finding_master_path(folder="data") :
    """just find our data folder in the repo structure from
    anywhere"""

    path = os.getcwd()
    path = path.split("/")

    idx  = path.index(FOLDER)
    path = path[:idx+1]
    folder = str(folder) + "/"
    path.append(folder)
    
    path = "/".join(path)

    # check if path is a valid path
    if not os.path.isdir(path) : 
        raise NotADirectoryError

    return path
    

# @timer
def return_datasets(path) : 

    li = [i for i in os.listdir(path) if ".csv" in i ]
    
    return li 


# @timer
def build_df(path, file) : 

    df          = pd.read_csv(path+file, index_col=0)
    df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            "target"], dtype="object")

    return df


# @timer
def print_df(df) : 

    print(df.ndim)
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    print(df.columns)
    print(df.describe())
    print(df.head(3))
    print(df.tail(3))


# @timer
def re_dtype(df) : 

    # li = [np.uint8, np.uint16]
    # [print(i,  np.iinfo(i).min, np.iinfo(i).max) for i in li]

    dtypes_dict = {     "last_don"  : np.uint8, 
                        "num_don"   : np.uint8,
                        "vol_don"   : np.uint16, 
                        "first_don" : np.uint8, 
                        "target"    : np.uint8       }


    df = df.astype(dtypes_dict)

    return df 


# @timer
def graph_each_feature(df)  : 

    features = [i for i in df.columns if "target" not in i] 

    fig, _axes = plt.subplots(2, 2, figsize=(13,13))
    axes = _axes.flatten()

    info(fig)
    info(axes)
    info(len(axes))

    for i, feat in enumerate(features) :
        info(i, feat)

        # -----------------------------------------
        # sns.distplot --> (kde=True ) ???
        # -----------------------------------------

        axes[i].hist(df[feat], bins=30)
        axes[i].set_title(feat)

    plt.suptitle("features distribution")
    
    plt.show()


# @timer
def graph_corr_matrix(df) : 

    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g')

    plt.title("correlation matrix")
    
    plt.show()


# @timer
def drop_corr_features(df) : 

    df = df.drop("vol_don", axis=1)

    return df 


# @timer
def study_nas(df) : 

    print(df.isna().any())
    print(df.isna().any())


# @timer
def study_outliers(df, k=1.5) : 

    fig, _axes = plt.subplots(1, 5, figsize=(13,13))
    axes = _axes.flatten()

    info(fig)
    info(axes)
    info(len(axes))

    for i, feat in enumerate(df.columns) :
        info(i, feat)

        axes[i].boxplot(df[feat], whis=k)
        axes[i].set_title(feat)

    plt.suptitle("features outliers, k of {}".format(whis))
    
    plt.show()


# @timer
def return_outliers(ser, k) : 

    desc = ser.describe()
    q1, q3, q2 = desc["25%"], desc["75%"], desc["50%"]
    IQ = q3-q1
    range_min, range_max = q1 - k * IQ, q3 + k*IQ

    # outliers = ser[(ser > range_max) or (ser < range_min)]
    
    return ser >= range_max


# @timer
def delete_outliers(df, k) : 

    li = [i for i in df.columns if "target" not in i]

    for feat in li : 
        df = df[return_outliers(df[feat], k) == False]

    return df


@timer
def first_tour(folder="data", file=TRAIN_FILE) : 

    # build data path
    path = finding_master_path(folder)
    # info(path)                            # UNCOMMENT IF NEEDED

    # just show dataset list
    # datasets = return_datasets(path)      # UNCOMMENT IF NEEDED
    # info(datasets)                        # UNCOMMENT IF NEEDED

    # build our df
    df = build_df(path, file)

    # print main info
    # print_df(df)                          # UNCOMMENT IF NEEDED

    # (overkilled) recast dataframe in a better dtype
    df = re_dtype(df)

    # graph features distr and correlation  # UNCOMMENT IF NEEDED
    # graph_each_feature(df)                  
    # graph_corr_matrix(df)                 # UNCOMMENT IF NEEDED

    # drop corr values
    df = drop_corr_features(df)

    # nas
    # study_nas(df)                         # UNCOMMENT IF NEEDED

    # for i in [1.5, 2, 2.5, 3] :           # UNCOMMENT IF NEEDED
    # study_outliers(df, i)                 # UNCOMMENT IF NEEDED

    # df = delete_outliers(df, 3)           # UNCOMMENT IF NEEDED

    return df

