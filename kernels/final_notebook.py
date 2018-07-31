#!/usr/bin/env python3
#-*- coding: utf8 -*-



################################################################################
################################################################################

# DRIVEN DATA - PREDICT BLOOD DONATIONS

################################################################################
################################################################################


# from IPython.display import Image
# Image("../intro.png")


################################################################################
################################################################################

# 00-first_dataset_tour.py

################################################################################
################################################################################


# Find here a first study of the dataset, in which we seek to understand and
# give meaning to the dataset.

# We are not trying to solve our problem but will focus on visualization,
# clenaning and feature engineering.

# At first we will just study the corelations, the links, the quality and the
# meaning of our dataset.External research and more general considerations may 
# be included in this work


# import

import os, sys, logging, random, time

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns


# logging and warnings

# logger = logging.getLogger()
# logger.setLevel(logging.CRITICAL)

l = logging.INFO
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

# import warnings
# warnings.filterwarnings('ignore')


# graph
%matplotlib
# get_ipython().magic('matplotlib inline')
sns.set()


# consts

PROJECT     = "Driven-Data-Blood-Donations"
DATA        = "data"
SUBMISSIONS = "submissions"
TRAIN_FILE  = "training_data.csv"
TEST_FILE   = "test_data.csv"


# 'overkill' First we define two decorators. it will be useful to control 
# complex and long algo (grid search, finding meta params...) :

def caller(funct) : 
    """decorator to give call and end of a function"""

    def wrapper(*args, **kwargs) : 

        msg = funct.__name__ + " : called"
        print(msg)
        

        res = funct(*args, **kwargs)

        msg = funct.__name__ + " : ended"
        print(msg)

        return res

    return wrapper


def timer(funct) : 
    """decorator to give runing time of a function"""

    def wrapper(*args, **kwargs) : 
        
        t = time.time()

        res = funct(*args, **kwargs)

        t = round(time.time() - t, 2)
        msg = funct.__name__ + " : " + str(t) + " secs" 
        print(msg)

        return res

    return wrapper


####


# @caller
# @timer
# def _print() : 
#     print("this is a test")
    
# _print()


# Our first function : just find the data folder in the repo 
# structure from anywhere

def finding_master_path(folder, project=PROJECT) :
    """just find our data folder in the repo structure from
    anywhere"""

    path = os.getcwd()
    path = path.split("/")

    idx  = path.index(project)
    path = path[:idx+1]
    folder = str(folder) + "/"
    path.append(folder)
    
    path = "/".join(path)

    # check if path is a valid path
    if os.path.isdir(path) : 
        return path
    
    li = [i for i in os.listdir() if (os.path.isdir(i) and (i[0]!="."))]
    if project in li : 
        path = os.getcwd + "/" + project
        return path

    return ValueError("project not found, please 'cd project")


####


# path = finding_master_path(DATA)
# path


# Control that our datafiles are Ok

def return_datasets(path) : 

    return [i for i in os.listdir(path) if ".csv" in i ]


####


# datasets = return_datasets(path)
# datasets


# Init our dataframe 

def init_df(path, file) : 

    df = pd.read_csv(path+file, index_col=0)

    if len(df.columns)  == 5 : 
        df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            "target"], dtype="object")
    elif len(df.columns )  == 4 : 
        df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            ], dtype="object")
    else : 
        raise ValueError("invalid numb of columns")

    return df


####


# df = init_df(path, TRAIN_FILE)
# df.head()


# Let's have a first raw tour about this df

def print_df(df) : 

    print("data frame dimension :       ")
    print(df.ndim)

    print("\n\ndata frame shape :       ")
    print(df.shape)

    print("\n\ndata frame types :      ")
    print(df.dtypes)

    print("\n\ndata frame index :       ") 
    print(df.index)

    print("\n\ndata frame columns :     ")
    print(df.columns)

    print("\n\ndata frame head :        ")
    print(df.head(3))

    print("\n\ndata frame tail :        ")
    print(df.tail(3))

    print("\n\ndata frame desc :        ")
    print(df.describe())

    
####


# print_df(df)


# 'overkill' Let's retype our values to reduce mem usage 

def re_dtype(df) : 

    # li = [np.uint8, np.uint16]
    # [print(i,  np.iinfo(i).min, np.iinfo(i).max) for i in li]

    if len(df.columns) == 5 : 
        dtypes_dict = { "last_don"  : np.uint8, 
                        "num_don"   : np.uint8,
                        "vol_don"   : np.uint16, 
                        "first_don" : np.uint8, 
                        "target"    : np.uint8       }

        return df.astype(dtypes_dict)

    elif len(df.columns) == 4 : 
        dtypes_dict = { "last_don"  : np.uint8, 
                        "num_don"   : np.uint8,
                        "vol_don"   : np.uint16, 
                        "first_don" : np.uint8,      }

        return df.astype(dtypes_dict) 

    raise ValueError("pb occured")
    

####


# df = re_dtype(df)
# df.head()
# df.dtypes


# Let's have a second tour of our dataset but with graphical tools

def graph_each_feature(df)  : 

    # features = [i for i in df.columns if "target" not in i] 

    features = df.columns

    fig, _axes = plt.subplots(1, 5, figsize=(13,13))
    axes = _axes.flatten()

    info(fig)
    info(axes)
    info(len(axes))

    for i, feat in enumerate(features) :
        info(i, feat)

        # -----------------------------------------
        # use sns.pairplot() !!!!!!
        # sns.distplot --> (kde=True ) ???
        # -----------------------------------------

        axes[i].hist(df[feat], bins=30)
        axes[i].set_title(feat)

    plt.suptitle("features distribution")
    plt.show()
    
    
####


# graph_each_feature(df)


# Idem for target

def graph_target(df) : 

    df.target.hist()
    plt.xlabel("target")
    plt.ylabel("values")
    plt.title("target value's distribution")
    plt.show()
    
    
####


# graph_target(df)


# Looking in depth : finding correlation between features

def graph_corr_matrix(df) : 
    
    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g')

    plt.title("correlation matrix")
    plt.show()

    
####


# graph_corr_matrix(df)


# So without doupt we can drop perfectly correlated features

def drop_corr_features(df) : 

    df = df.drop("vol_don", axis=1)

    return df 


####


# df = drop_corr_features(df)
# df.head()


# What about nas?

def study_nas(df) : 

    print("nas all : ")
    print(df.isna().all())
    print("\n\nnas any : ")
    print(df.isna().any())
    
    
####


# study_nas(df)


# Ok perfect, too easy maybe...
# what about outliers ? 

def study_outliers(df, k=1.5) : 

    fig, _axes = plt.subplots(1, 4, figsize=(13,13))
    axes = _axes.flatten()

    info(fig)
    info(axes)
    info(len(axes))

    for i, feat in enumerate(df.columns) :
        info(i, feat)
        axes[i].boxplot(df[feat], whis=k)
        axes[i].set_title(feat)

    plt.suptitle("features outliers for k > {}".format(k))
    
    plt.show()

    
####


# study_outliers(df)


# After all ! we have a first real data scientist job to do : cleaning! 
# so we will design one function to be able to reshape our df at will

def return_outliers(ser, k) : 

    desc = ser.describe()
    q1, q3, q2 = desc["25%"], desc["75%"], desc["50%"]
    IQ = q3-q1
    range_min, range_max = q1 - k * IQ, q3 + k*IQ

    # outliers = ser[(ser > range_max) or (ser < range_min)]
    
    return ser >= range_max


def delete_outliers(df, k) : 

    li = [i for i in df.columns if "target" not in i]

    for feat in li : 
        df = df[return_outliers(df[feat], k) == False]

    return df


####


# print(df.shape)
# _df = delete_outliers(df, 1.5)
# print(_df.shape)


# Let's resume all of this in a global function

# @caller
# @timer
def first_tour(folder=None, filename=None, project=PROJECT) : 

    # build data path
    path = finding_master_path(folder, project)
    print(path)                             # UNCOMMENT IF NEEDED

    # just show dataset list
    datasets = return_datasets(path)        # UNCOMMENT IF NEEDED
    print(datasets)                         # UNCOMMENT IF NEEDED

    # build our df
    df = init_df(path, filename)

    # print main info
    print_df(df)                            # UNCOMMENT IF NEEDED

    # (overkilled) recast dataframe in a better dtype
    df = re_dtype(df)

    # graph features distr and correlation  # UNCOMMENT IF NEEDED
    graph_each_feature(df)                  
    graph_corr_matrix(df)                   # UNCOMMENT IF NEEDED

    # drop corr values
    df = drop_corr_features(df)

    # nas
    study_nas(df)                           # UNCOMMENT IF NEEDED

    for i in [1.5, 2, 2.5, 3] :             # UNCOMMENT IF NEEDED
        study_outliers(df, i)               # UNCOMMENT IF NEEDED


# Finally we define a function to auto build our data frame

@caller
@timer
def build_df(folder=None, filename=None, project=PROJECT) : 

    path = finding_master_path(folder, project)
    df = init_df(path, filename)
    df = re_dtype(df)
    df = drop_corr_features(df)

    return df


####


df = build_df(DATA, TRAIN_FILE)
# df.head()


# Conclusion

# Through this first study, we can see several things. Our dataset is of very
#  good quality, few outliers, 
# no missing values, a number of features reduced and little corelation.

# Its simplicity will be an obvious limit when it comes to making feature 
# engineering, benchmarking 
# models and looking for marginal improvements.


################################################################################
################################################################################

# 01-first_naive_models.py

################################################################################
################################################################################


# In this second part, we will implement our first logistic regression model.

# We will first implement by hand a naive classifier, then a dummy classifier 
# (who does the same job), and finally a basic logistic regression model.

# Rather than looking at the results of a regression we will implement a 
# function that will test the model x times and that will average the results
# obtained


# import

from sklearn.model_selection import train_test_split

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV

# from first_dataset_tour import *


# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# Split our features from our target

def return_X_y(df) : 

    if "target" in df.columns : 

        X = df.drop("target", axis=1)
        y = df.target

        return X, y  

    else : 
        return df 

    
####


# X,y = return_X_y(df)
# print(X.columns)
# print(y.name)


# Split test and train df/target

def split(X,y, size=0.33) : 

    return train_test_split(X, y, test_size=size)


####


# X_tr, X_te, y_tr, y_te = tup = split(X,y)
# for i in tup : print(i.shape)
    


# 'overkill' Build from scratch a naive/dummy model which make prediction 
# regarding global target probabilities

def naive_model(df=None, param=None) :

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test  = split(X,y)

    freq = y_test.value_counts() / len(y_test)
        
    y_pred = np.random.binomial(1, freq[1], len(y_test))
    y_pred = pd.Series(y_pred)

    lolo = log_loss(y_test, y_pred).round(3)

    return lolo, None 


####


# lolo, mod = naive_model(df)
# print(lolo)


# Rather than coding a dummy model from scratch, use sklearn DummyClassifier 
# (same job)

def dummy_model(df=None, param=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)
    
    X,y     = return_X_y(df)
    X_train, X_test, y_train, y_test = split(X,y)

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test) 

    lolo = log_loss(y_test, y_pred).round(3)

    return lolo, model


####


# lolo, mod = dummy_model(df)
# print(lolo)


# Just for fun trying to make predictions with a very basic model (no meta 
# params, no features engineering) this one will be our model prediction base
# it is suposed to be better that our DummyClassifier. If not there is a major
# issue...

def basic_model(df=None, param=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)
    
    X,y     = return_X_y(df)

    X_train, X_test, y_train, y_test = split(X,y)

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    y_pred = y_pred[:, 1]

    lolo = log_loss(y_test, y_pred).round(3)
    
    return lolo, model


####


# lolo, mod = basic_model(df)
# print(lolo)


@caller
@timer
def model_accuracy_mean(model=None, n=50, df=None) : 
    
    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = basic_model

    scores = pd.Series([model(df)[0] for i in range(n)])

    fix, ax = plt.subplots(1,2)
    scores.hist(ax=ax[0])
    pd.DataFrame({"":scores}).boxplot(ax=ax[1])
    plt.suptitle("log loss hist and box plot for {} runs for basic_model".format(n))
    plt.show()
    
    info(type(scores))
    info(type(range(n)))

    return scores.describe()


####


# model_accuracy_mean(n=10)



@caller
@timer
def first_approch_of_feat_eng(model=None, n=20, df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = basic_model
        
    k_list = np.arange(10, 50)/10
    
    outlier_df = lambda k : delete_outliers(df, k)
    results = [ [basic_model(outlier_df(k))[0] for k in k_list]
                   for i in range(n) ]
    
    results = pd.DataFrame(results, columns=k_list)
       
    results.boxplot()
    # plt.x_label("k") ; plt.y_label("log loss")
    plt.title("log loss for various 'k' values - outliers detection")
    plt.show()
    
    return results
    
    
####

# results = first_approch_of_feat_eng(n=10)

# raw data
# results.iloc[:10, :]


# fancy results
# results.describe().T.sort_values(by="50%", axis=0).iloc[:10, :]

# graph 
# results.iloc[:10, :].T.plot()


# without getting mad about results, we can see that it is not very easy to use
# maybe it could be much better if we only look at gain between normal and 
# ehanced results

def outliers_lolo_gain(k, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model :  model = LogisticRegression()
        
    if not params : params = dict()

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")
        
    # init lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    init_lolo = lolo = log_loss(y_te, y_pred)

    # new lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(delete_outliers(df, k)))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    new_lolo = log_loss(y_te, y_pred)
    
    return round((init_lolo - new_lolo) / init_lolo,3)


# and here we have our benchmark function which return log loss gain for 
# each k value

def benchmark_various_outliers( n=20, df=None, params=None, 
                                        model=None, outliers_list=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    if not outliers_list : 
        outliers_list = np.arange(1,5.01,0.1).round(1)

    results = [[outliers_lolo_gain(k, df, model, params) for k in outliers_list]
                     for _ in range(n)]
    
    results = pd.DataFrame(results, columns=outliers_list)
    
    return results


####

# results = benchmark_various_outliers(10)

# raw results 
# results.iloc[:10, :]



# how to have global stats ?

def print_gobal_stats(results) : 

    print("mean      : ", round(results.mean().mean(),3))
    print("median    : ", round(results.median().median(),3))
    print("25%       : ", round(results.quantile(0.25).quantile(0.25),3))
    print("75%       : ", round(results.quantile(0.75).quantile(0.75),3))
    print("min       : ", round(results.min().min(),3))
    print("max       : ", round(results.max().max(),3))
    
    
####


# print_gobal_stats(results)

# graph 
# results.boxplot()

# fancy results
# results.describe().T.sort_values(by="50%", ascending=False).iloc[:10, :]

# graph 
# results.iloc[:10, :].T.plot()

# last graph 
# results.describe().T.loc[:,["mean", "50%"]].plot()


# can we say that outliers detection and cleaning has a significant impact 
# regarding our final score? 
# the answer is of course : "No"
# but we can try with a k=1.3 outlier... 

n       = 100
results = pd.Series([outliers_lolo_gain(1.4, df) for i in range(n)])


# results.describe().T


def enhanced_describe(results): 
    
    _results = results.describe()
    _min, Q1, _med, Q3, _max = (    _results["min"], _results["25%"], 
                                    _results["50%"], _results["75%"], 
                                    _results["max"]     )
    _mean = _results["mean"]
    _count = _results["count"]
    _std = _results["std"]
    
    Q1_Q3_vs_med = round((Q3 - _med) - (_med -Q1),3)
    ma_mi_vs_med = round((_max - _med) - (_med -_min),3)
    
    ind  = 10
    conf = 0
    
    if (round(_results["mean"],3) >0)  : conf += 1
    if (round(_results["mean"],3) <0)  : conf -= 1  
        
    if (round(_results["50%"],3) >0)  : conf += 1
    if (round(_results["50%"],3) <0)  : conf -= 1    
        
    if ( (round(_results["50%"],3) >0) and (round(_results["mean"],3) >0)) : conf += 1
    if ( (round(_results["50%"],3) <0) and (round(_results["mean"],3) <0)) : conf -= 1
    
    if round((Q3 - _med) - (_med -Q1),3) >0 : conf += 1
    if round((Q3 - _med) - (_med -Q1),3) <0 : conf += 1

    if round((_max - _med) - (_med -_min),3) >0 : conf += 1
    if round((_max - _med) - (_med -_min),3) <0 : conf -= 1
        
    if ( (round((_max - _med) - (_med -_min),3) >0) 
        and (round((Q3 - _med) - (_med -Q1),3) >0)  ) :
            conf += 1
    if ( (round((_max - _med) - (_med -_min),3) <0) 
       and (round((Q3 - _med) - (_med -Q1),3) <0)  ) :
            conf -= 1                                         

    if _min > 0 : count += 2
    if _max < 0 : count -= 2
    
    if Q1 > 0  :  count += 2
    if Q3 < 0  :  count += 2
        
    conf = round(conf /ind ,2)
    strength = pd.Series( [ _med/0.1, _mean/0.1, Q1_Q3_vs_med/0.1, ma_mi_vs_med/0.1]).mean()
            
    sol = _count/100
    
    output = pd.Series([    _count, _med, _mean, _std, _min, _max, Q1, 
                            Q3, Q3-Q1, Q1_Q3_vs_med,
                            ma_mi_vs_med, sol, conf, strength], 
                            index = [ "count", "med", "mean", "std", "min", 
                                      "max", "Q1", "Q3", "IQ", "IQ vs med", 
                                      "extr vs med", "sol", "conf", "strength"])
    
    output = output.round(3)
    output = pd.DataFrame({"Enhanced describe" : output}, index=output.index)

    # graph 
    fig, axes = plt.subplots(1,2)
    axes[0].boxplot(results)
    axes[0].scatter(1, results.mean())
    pd.DataFrame(results).hist(ax=axes[1])

    return output
    
####


# enhanced_describe(results)


# conclusion

# We could see through this first study that we have a dataset quite simple, 
# allowing our first approach 
# to have pretty good results.

# Our base model offers a performance of 0.5 and the impact of the ouliers 
# on the model's performance 
# seems at first glance quite low.

# Having a fairly simple dataset, the possibilities for improving the models 
# will not be that simple due 
# to the small number of variables and the small size of the dataset : the 
# possibilities offered by 
# the feature engineering are indeed quite low.


################################################################################
################################################################################

# 02-starting_ML.py

################################################################################
################################################################################


# In this third part we will finally start to make real machine learning. We will first code a high-level
# function to handle the tests of the different models.

# We will then benchmark the different classification models as well as the impact of the different meta 
# variables on the relevance of the basic model: number of folds, preprocessing, scoring method, clips
# of the predicted values, etc.

# This work is clearly laborious, but its successful execution depends on our ability to really push 
# our model at best.


# import

from math import ceil
import itertools as it
from collections import OrderedDict, Iterable

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

# from first_naive_models import *


# let's define a new decorator that will allow us to beeper during long tests of algo sessions

def beeper(funct) : 
    """decorator to beep when a long algo as finished"""

    def wrapper(*args, **kwargs) : 
        
        res = funct(*args, **kwargs)

        freq=440
        duration=3
        cmd = 'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq)
        os.system(cmd)

        return res

    return wrapper


# Please find here our meta GridSearchCV. For each params grid, it initiates a df, split 
# test / train data, launches the grid search, and returns the model and the log loss in final output

# @timer
# @caller
def run_GSCV(   model=None,     params=None, 
                df=None,        tup=None,
                cv=0,           n_jobs=None,
                scoring=None,   test_size=None, 
                verbose=None,   debug_mode=None,
                clip=None) : 

    # init default params
    if not model        : model = LogisticRegression()
    else : 
        try             : model = model()
        except          : pass

    is_df               = isinstance(df, pd.DataFrame)
    if not is_df        : df = build_df(DATA, TRAIN_FILE)
    if not params       : params = dict()
    if not cv           : cv=10
    if not n_jobs       : n_jobs = 6
    if not scoring      : scoring = "average_precision"
    if not verbose      : verbose = 1
    if not test_size    : test_size = 0.33
    if not debug_mode   : debug_mode = False
    grid = None

    if not tup : # prepare X, y
        X,y                 = return_X_y(df)
        X_tr,X_te,y_tr,y_te = split(X,y, test_size)
    else :
        X_tr,X_te,y_tr,y_te = tup
    
    info(model.__class__)
    try : # init grid      
        grid        = GridSearchCV( estimator=model, 
                                    param_grid=params,  
                                    cv=cv, 
                                    n_jobs=n_jobs,
                                    scoring=scoring, 
                                    verbose=verbose)
        info("grid init ok")
        info(grid)

    except Exception as e : 
        info("grit init went wrong")
        if debug_mode : input()
        raise(e)

    try : # fit
        grid.fit(X_tr, y_tr)
        info("grid fit OK")
        info(grid.best_estimator_)
        info(grid.best_score_)
        info(grid.best_params_)

    except Exception as e : 
        info("grit fit went wrong")
        if debug_mode : input()
        raise(e)

    try : # pred
        y_pred = grid.predict_proba(X_te)
        y_pred = y_pred[:,1]
        info("pred OK")
        info("run_GSCV 0") 
    
    except Exception as e : 
        info("pred went wrong")
        info("maybe predict_proba do not exists just predict")       
        try : 
            y_pred = grid.predict(X_te)
            info("second pred Method OK")
            info("run_GSCV 1")

        except Exception as e : 
            info("2nd pred method went wrong")
            if debug_mode : input()
            raise(e)

    try : # compute log_loss as 'lolo' 
        if clip : 
            y_pred = clipping_log_loss(y_pred, x=clip)
        lolo = log_loss(y_te, y_pred).round(3)
        info("lolo ok")
        info(lolo)

    except Exception as e : 
        info("lolo went wrong")
        if debug_mode : input()
        raise(e)

    # return lolo and grid
    if isinstance(lolo, float) and grid : 
        return lolo, grid
    # else raise Error
    raise ValueError("lolo and/or grid error")
    
    
####

# lolo, grid = run_GSCV()

# print(lolo)
# print(grid)


# Find here the models we will try and test

COLUMNS = [     "LR",       "RC",
                "SVC",      # "Nu",
                "KNN",
                "DT", 
                "RF", 
                "Ada", 
                "Per",      "MLP"   ]

MODELS = [      LogisticRegression, RidgeClassifier,
                LinearSVC, # NuSVC,
                KNeighborsClassifier,
                DecisionTreeClassifier,
                RandomForestClassifier,
                AdaBoostClassifier,
                Perceptron, MLPClassifier   ]


# we now will benchmark various models, without any feat eng. or meta params. Just have a first look...

# @timer
def benchmark_various_models(  n=5, df=None, graph=True, params=None,
                                    models = MODELS, columns= COLUMNS) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    if not params : params = dict()    

    results = [     pd.Series([run_GSCV(m, params, df)[0] for m in models], 
                        index=columns) for i in range(n)]
    
    results = pd.DataFrame(results, columns=columns)

    return results


####


# results = benchmark_various_models(10)


# print out raw values
# results.iloc[:10, :]

# lets have fancy representation of our results
# _results = results.describe().T.sort_values(by="50%")
# _results


# graph it 
# results.boxplot()
# plt.xlabel("models")
# plt.ylabel("log_loss score")
# plt.title("benchmark various models, without feat eng or meta params")
# plt.show()

# results = benchmark_various_outliers(10)

# raw results
# results.iloc[:10, :]

# global stats
# _results = results.describe().T
# _results.iloc[:10, :]

# graph 
# results.boxplot()
# plt.xlabel("outliers 'k' values")
# plt.ylabel("log_loss score")
# plt.title("benchmark various outliers 'k' values, without feat eng or meta params")
# plt.show()

# graph
# results.describe().T.loc[:, ["mean", "50%"]].plot()
# plt.xlabel("outliers 'k' values")
# plt.ylabel("log_loss score")
# plt.title("benchmark various outliers 'k' values, without feat eng or meta params")
# plt.show()

# graph
# results.T.iloc[:, :10].plot()
# plt.xlabel("outliers 'k' values")
# plt.ylabel("log_loss score")
# plt.title("benchmark various outliers 'k' values, without feat eng or meta params")
# plt.show()


# Now we will study various features transformation.
# First let's define some useful function

def transform_df(Tool, df=None, **kwargs) :

    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if "target" in df : 
        X = df.drop("target", axis=1)
        y = df.target
    else : 
        X = df

    tool = Tool()
    _X = tool.fit_transform(X)
    _X = pd.DataFrame(_X, columns=X.columns, index=X.index)
    
    if "target" in df :
        _df = _X
        _df["target"] = y
    else : 
        _df = _X

    return _df


def nothing(df=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    return df


def standscale(df=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    return transform_df(StandardScaler, df)



def normalize(df=None, norm=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not norm : 
        norm='l2'

    return transform_df(StandardScaler, df) 
    

minmax                 = lambda df : transform_df(MinMaxScaler, df)

minmax_and_standscale  = lambda df : standscale(minmax(df))

standscale_and_minmax  = lambda df : minmax(standscale(df))

minmax_and_normalize   = lambda df : normalize(minmax(df))
    
normalize_and_minmax   = lambda df : minmax(normalize(df))


# transorm list and index (str values)

TRANSFORM_LIST  = [  nothing, normalize, standscale, minmax, minmax_and_standscale, standscale_and_minmax, 
                     minmax_and_normalize, normalize_and_minmax]
TRANSFORM_INDEX = [  "nothing", "normalize", "standscale", "minmax", "minmax+stdsca", "stdsca+minmax", 
                     "minmax+norm", "minmax+norm"]



def transform_lolo_gain(method, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model :  model = LogisticRegression()
        
    if not params : params = dict()

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")
        
    # init lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    init_lolo = lolo = log_loss(y_te, y_pred)

    # new lolo
    df = method(df)
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    new_lolo = log_loss(y_te, y_pred)
    
    return round((init_lolo - new_lolo) / init_lolo,3)


# here we have our benchmark function 

def benchmark_various_transform(   n=10, df=None, graph=True, params=None, model=None,
                                    transform_list=TRANSFORM_LIST, transform_index = TRANSFORM_INDEX) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not model : model =  LogisticRegression()

    if not params : params = dict() 

    if len(transform_list) != len(transform_index) : 
        raise ValueError("lens not goods")

    results = [ [transform_lolo_gain(transf, df, model, params) for transf in transform_list]
                       for i in range(n)]
    
    results = pd.DataFrame(results, columns=transform_index)

    return results


####


# results = benchmark_various_transform(30)


# raw results 
# results.iloc[:10, :]

# gobal results
# print_gobal_stats(results)
    
# graph  
# results.boxplot()
# plt.xlabel("transformations of df")
# plt.ylabel("log_loss score")
# plt.title("benchmark various df transforms, without feat eng or meta params")
# plt.show()

# fancy results
# _results = results.describe().T.sort_values(by="50%", ascending=False)
# _results

# graph 
# results.T.iloc[:,:10].plot()
# plt.xlabel("transformations of df")
# plt.ylabel("log_loss score")
# plt.title("benchmark various df transforms, without feat eng or meta params")


def scoring_lolo_gain(scoring, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model :  model = LogisticRegression()
        
    if not params : params = dict()

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")
        
    # init lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    init_lolo = lolo = log_loss(y_te, y_pred)

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring=scoring)
    
    # new lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    new_lolo = log_loss(y_te, y_pred)
    
    return round((init_lolo - new_lolo) / init_lolo,3)


# ok let's do the same thing for scoring ! 

def benchmark_various_scoring(  n=5, df=None, graph=True, params=None, model=None,
                                scoring_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not scoring_list : 
        scoring_list = ['accuracy', 'neg_log_loss', 'f1', 'average_precision', 'precision', 'recall', "roc_auc"]

    if not model : model = LogisticRegression()

    if not params : params = dict() 

    results = [ [scoring_lolo_gain(s, df, model, params)  for s in scoring_list] for i in range(n)]
    
    results = pd.DataFrame(results, columns=scoring_list)

    return results


####

# results = benchmark_various_scoring(10)


# raw results
# results.iloc[:10, :]

# gobal results
# print_gobal_stats(results)

# graph 
# results.boxplot()
# plt.xlabel("scoring methods for grid search")
# plt.ylabel("log_loss score")
# plt.title("benchmark various scoring, without feat eng or meta params")
# plt.show()

# fancy results
# _results = results.describe().T.sort_values(by="50%", ascending=False)
# _results

# graph 
# results.T.iloc[:, :10].plot()
# plt.xlabel("scoring methods for grid search")
# plt.ylabel("log_loss score")
# plt.title("benchmark various scoring, without feat eng or meta params")
# plt.show()


def cv_lolo_gain(cv, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model :  model = LogisticRegression()
        
    if not params : params = dict()

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")
        
    # init lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    init_lolo = lolo = log_loss(y_te, y_pred)

    grid = GridSearchCV(model, params, cv = cv, n_jobs=6, scoring="accuracy")
    
    # new lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    new_lolo = log_loss(y_te, y_pred)
    
    return round((init_lolo - new_lolo) / init_lolo,3)


# 'overkill' OK, let's be evil, we now are ready to benchmark Kfolds numbers and test/train size ! 

def benchmark_various_cv(   n=5, df=None, graph=True, params=None, model=None,
                            cv_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not cv_list : 
        cv_list = [int(i) for i in np.arange(2,11, 1)]

    if not model : 
        model = LogisticRegression()

    if not params : params = dict() 

    results = [ pd.Series([cv_lolo_gain(c, df, model, params)  for c in cv_list], 
                        index=cv_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=cv_list)

    return results


####

# results = benchmark_various_cv(10)


# raw results
# results.head()

# gobal results
# print_gobal_stats(results)

# graph
# results.boxplot()
# plt.xlabel(" nb of kfolds for grid search")
# plt.ylabel("log_loss score")
# plt.title("benchmark various nb of kfolds, without feat eng or meta params")
# plt.show()

# fancy results
# _results = results.describe().T.sort_values(by="50%", ascending=False)
# _results

# graph 
# results.T.iloc[:, :10].plot()
# plt.xlabel("nb of kfolds methods for grid search")
# plt.ylabel("log_loss score")
# plt.title("benchmark various nb of kfolds, without feat eng or meta params")
# plt.show()


# without getting mad about results, we can see that it is not very easy to use
# maybe it could be much better if we only look at gain between normal and ehanced results

def test_train_lolo_gain(test_size, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model :  model = LogisticRegression()
        
    if not params : params = dict()

    grid = GridSearchCV(model, params, cv=3, n_jobs=6, scoring="accuracy")
        
    # init lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df))  
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    init_lolo = lolo = log_loss(y_te, y_pred)

    # new lolo
    X_tr, X_te, y_tr, y_te = split(*return_X_y(df), size=test_size)
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)[:, 1]
    new_lolo = log_loss(y_te, y_pred)
    
    return round((init_lolo - new_lolo) / init_lolo,3)


# 'overkill' idem for test/train ratio

def benchmark_various_test_size(    n=5, df=None, graph=True, params=None, 
                                    model=None, test_size_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not test_size_list : 
        test_size_list =  [round(float(i), 2) for i in (np.arange(20, 41, 1) /100)]

    if not model : 
        model = LogisticRegression()

    if not params : params = dict() 

    results = [ pd.Series([test_train_lolo_gain(t, df, model, params) 
                        for t in test_size_list], 
                        index=test_size_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=test_size_list)

    return results


#### 

# results = benchmark_various_test_size(10)


# raw results
# results.iloc[:10, :]

# gobal results
# print_gobal_stats(results)

# graph 
# results.boxplot()
# plt.xlabel("nb of test_size for test/train split")
# plt.ylabel("log_loss score")
# plt.title("benchmark various test_size for test/train split, without feat eng or meta params")
# plt.show()

# fancy results
# _results = results.describe().T.sort_values(by="50%", ascending=False).iloc[:10, :]
# _results

# graph 
# results.T.iloc[:, :10].plot()
# plt.xlabel("nb of test_size for test/train split")
# plt.ylabel("log_loss score")
# plt.title("benchmark various test_size for test/train split, without feat eng or meta params")
# plt.show()


# we now will try to ehance our output with thresholding our predictions 

def clipping_log_loss(y_pred, y_test=None, x=0.05) : 
    
    if not isinstance(y_pred, Iterable) : 
        raise ValueError("y_pred has to be a pd.Series")
        
    if isinstance(y_test, pd.Series) :  
        info(log_loss(y_test, y_pred))
    
    if not(0.000 <= x <= 0.4999 ) :
        raise ValueError("threshold must be 0.00 --> 0.5")

    info(x)
 
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.apply(lambda i : x if i<= x else i)
    
    x = round((1-x), 2)
    y_pred = y_pred.apply(lambda i : x if i>= x else i)

    if isinstance(y_test, pd.Series) :  
        info(log_loss(y_test, y_pred))
    
    return y_pred


# compute lolo gain for one k threshold

def clipping_lolo_gain(k, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    X,y = return_X_y(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    y_test = y_te
    
    grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)
    y_pred = y_pred[:, 1]

    init_lolo = lolo = log_loss(y_test, y_pred)

    y_pred = clipping_log_loss(y_pred, x=k)
    new_lolo = log_loss(y_test, y_pred)

    return round((init_lolo - new_lolo) / init_lolo,3)



# idem for every threshold between 0.0 and 0.5

def benchmark_various_lolo_clipping(   n=20, df=None, graph=True, params=None, 
                                        model=None, threshold_list=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    if not threshold_list : 
        threshold_list = np.arange(0.01,0.4, 0.01).round(2)
        # threshold_list = [round(i/1000, 3) for i in range(10,101)]
        # threshold_list = [round(i/1000, 3) for i in range(10,500, 5)]
    results = [ [clipping_lolo_gain(k, df, model, params) for k in threshold_list]
                     for _ in range(n)]
    
    results = pd.DataFrame(results, columns=threshold_list)
    
    return results


####

# results = benchmark_various_lolo_clipping(10)


# raw results
# results.iloc[:10, :]

# gobal results
# print_gobal_stats(results)

# graph 
# results.boxplot()

# fancy results 
# _results = results.describe().T.sort_values(by="50%", axis=0, ascending=False).iloc[:10, :]
# _results

# graph
# results.iloc[:10, :].T.plot()

# we now will study the impact of softmax method on our output predictions
# famous soft max method is the exponetial soft max

def exp_softmax(x):
    

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# we will compute our method, which is a linear soft max method

def lin_softmax(ser, k=2, p=0.1, t=0.5) : 
    
    is_u = lambda x : True if x >t else False
    dist_med = lambda x : abs(x -t) / t

    coef = lambda x :  p * (k * (t-dist_med(x)))

    _soft_extr = lambda x : x if dist_med(x)>= t else (x + coef(x) if is_u(x) else x - coef(x))

    return [_soft_extr(i) for i in ser]


# compute lolo gain for one k threshold

def softmax_lolo_gain(k, p, t, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    X,y = return_X_y(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    y_test = y_te
    
    grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)
    y_pred = y_pred[:, 1]

    init_lolo = lolo = log_loss(y_test, y_pred)

    y_pred = lin_softmax(y_pred, k, p, t)
    new_lolo = log_loss(y_test, y_pred)

    return round((init_lolo - new_lolo) / init_lolo,3)


def benchmark_various_softmax_lolo(   n=20, df=None, graph=True, params=None, 
                                        model=None, softmax_list=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    if not softmax_list : 

        t = 0.5

        k = np.arange(1,5.1, 0.5).round(2)
        p = np.arange(0.1, 0.51, 0.05).round(2)
        softmax_list = it.product(k, p)
        softmax_list = [i for i in softmax_list]
    
    results = [ [softmax_lolo_gain(k, p, t, df, model, params) for k,p in softmax_list]
                     for _ in range(n)]
    
    results = pd.DataFrame(results, columns=softmax_list)
    
    return results


####


# results = benchmark_various_softmax_lolo(10)
# results.to_csv("benchmark_various_softmax_lolo1.csv", index=False)


# raw results
# results.iloc[:10, :]

# gobal results
# print_gobal_stats(results)

# fancy results 
# _results = results.describe().T.sort_values(by="50%", axis=0, ascending=False).iloc[:10, :]
# _results


# compute lolo gain for one k threshold

def softmax_clip_lolo_gain(x, k, p, t, df, model=None, params=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    X,y = return_X_y(df)
    X_tr, X_te, y_tr, y_te = split(X, y)
    y_test = y_te
    
    grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="accuracy")
    
    grid.fit(X_tr, y_tr)   
    y_pred = grid.predict_proba(X_te)
    y_pred = y_pred[:, 1]

    init_lolo = lolo = log_loss(y_test, y_pred)

    y_pred = lin_softmax(y_pred, k, p, t)
    y_pred = clipping_log_loss(y_pred, x=x)
    new_lolo = log_loss(y_test, y_pred)

    return round((init_lolo - new_lolo) / init_lolo,3)



def benchmark_various_softmax_clipping_lolo(    n=20, df=None, graph=True, 
                                                params=None, 
                                                model=None, 
                                                softmax_clip_list=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    if not softmax_clip_list : 

        t = 0.5
        x = np.arange(0.01, 0.11, 0.01).round(2)
        k = np.arange(1,5.1, 0.5).round(2)
        p = np.arange(0.1, 0.51, 0.1).round(2)
        softmax_clip_list = [ (a,b,c) for a in x for b in k for c in p]
        
        if len(softmax_clip_list) >= 500 : 
            raise ValueError(" to long to compute")
    
    results = [ [softmax_clip_lolo_gain(x, k, p, t, df, model, params) for
                     (x, k, p) in softmax_clip_list] for _ in range(n)]
    
    results = pd.DataFrame(results, columns=softmax_clip_list)
    
    return results


####


# results = benchmark_various_softmax_clipping_lolo(10)
# results.to_csv("benchmark_various_softmax_clipping_lolo0.csv", index=False)  


# Conclusion

# After a full study of the different possibilities we have, here are the conclusions:

# - 3 models are very good, LogisticRegression, AdaBoost, and MLP (neural networks).
# - for obvious reasons we will first investigate LogisticRegression model
# - outlier threshold and tranform of the dataset does not really improve our score but allows for more 
#   consistent results and can outperform punctually
# - roc_auc and average_precision allow a slightly better score
# - Kfolds number, test/train rate have na 'real' on the global performance
# - without knowning precisily the good value, we can say that cliping our results is a good thing 
# - what about soft max ? it is not for now a good solution, but maybe we can try it later, with over
# params...


#####################################################################################################
#####################################################################################################

# 03-finding_good_models.py

#####################################################################################################
#####################################################################################################


# In this part we will concentrate our efforts on finding the most effective meta parameters 
# for the 3 best models.

# We will have to define a parser of parameter which will make it possible to test all the possible
# parameters by passing over the errors related to the incompatibility of certain combinations

# for each model it will be necessary to evaluate if it is 'good' and to evaluate in what way it is 
# better than a model by default


# import

# from starting_ML import *



def combine_param_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


####

# d = {"a" : ["a","b","c"], "b": [0,1,2,3,4]}
# d = combine_param_dict(d)
# d


def grid_LogisticRegression(df=None, param=None,
                            model=LogisticRegression) : 

    default_params  = { "penalty":["l2"],
                        "dual":[False],
                        "tol":[0.0001],
                        "C":[1.0],
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["liblinear"],
                        "max_iter":[100],
                        "multi_class":["ovr"],
                        "warm_start":[False],   }

    all_params      = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001, 0.001, 0.1, 1],                   # consider also np.logspace(-6, 2, 9)
                        "C":[0.0001, 0.001, 0.01, 0.1, 1, 10, 100],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100, 1000],   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[False, True],   }

    all_params2     = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001, 0.001, 0.01],            # consider also np.logspace(-6, 2, 9)
                        "C":[0.001, 0.01, 0.1, 1, 10],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],                   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[True, False],   }

    none_params     = {}

    if not param :  param = none_params
    else  :         param = best_params_2

    # lolo, grid       = run_GSCV(model, param, None)

    # return lolo, grid





# @decor_grid
# @timer
def grid_MLPClassifier( df=None, param=None,
                        model=MLPClassifier):

    default_params  = {}

    none_params     = {}
        
# "activation": ["identity", "logistic", "tanh", "relu"],   
# "solver": ["lbfgs", "sgd", "adam"],   
# "alpha": np.logspace(-4, 2, 9), 

    params          = { "hidden_layer_sizes": [(3,3,3), (3,3), (4,4), (5,5), (5,5,5), (4,4,4)],   
                        "activation": ["identity", "logistic", "tanh", "relu"],   
                        "solver": ["lbfgs", "sgd", "adam"],   
                        "alpha": np.logspace(-4, 2, 9),   
                        "batch_size": ["auto"],   
                        "learning_rate": ["constant", "invscaling", "adaptive"],   
                        "learning_rate_init": [0.001],   
                        "power_t": [0.5],   
                        "max_iter": [200, 1000, 5000],   
                        "shuffle": [True],      
                        "tol": [0.0001],      
                        "warm_start": [False, True],   
                        "momentum": [0.9],   
                        "nesterovs_momentum": [True],   
                        "early_stopping": [False],   
                        "validation_fraction": [0.1],   
                        "beta_1": [0.9],   
                        "beta_2": [0.999],   
                        "epsilon": [1e-08]}

    all_params      = { "hidden_layer_sizes": [(4,4,4)],   
                        "activation": ["identity", "logistic", "tanh", "relu"],   
                        "solver": ["lbfgs", "sgd", "adam"],   
                        "alpha": np.logspace(-4, 2, 9),   
                        "batch_size": ["auto"],   
                        "learning_rate": ["constant", "invscaling", "adaptive"],   
                        "learning_rate_init": [0.001],   
                        "power_t": [0.5],   
                        "max_iter": [200],   
                        "shuffle": [True],      
                        "tol": [0.0001],      
                        "warm_start": [True],   
                        "momentum": [0.9],   
                        "nesterovs_momentum": [True],   
                        "early_stopping": [False],   
                        "validation_fraction": [0.1],   
                        "beta_1": [0.9],   
                        "beta_2": [0.999],   
                        "epsilon": [1e-08]}



    if not param :  param = none_params
    else  :         param = params

    # lolo, grid       = run_GSCV(model, param, None)

    # return lolo, grid




all_params     =      { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001, 0.001],            # consider also np.logspace(-6, 2, 9)
                        "C":[ 0.01, 0.1, 1, 10],      # consider also np.logspace(-3, 1, 40)
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],                   # consider also np.logspace(3, 5, 3)
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[True, False],   }








def _mean(x) : 
    
    if not isinstance(x, Iterable) : 
        raise ValueError("x must be iter")

    return round(float(sum(x) / len(x)), 3)


def _med(x) : 

    x = sorted(x)

    if not (len(x) % 2) : 
        idx     = len(x) /2
        idx_u   = ceil(idx)
        idx_d   = ceil(idx) - 1

        med = _mean([x[idx_u], x[idx_d]])

    else :
        idx = int(len(x)/2)
        med = x[idx]

    return round(med, 3)


def _mix(x) : 

    mea_x = _mean(x)
    med_x = _med(x)

    return _mean([mea_x, med_x]) 


####

# ser = pd.Series([1,2,3,4,5,6,7,8, 10, 1000])
# print(ser.mean())
# print(ser.median())
# print(pd.Series([ser.mean(), ser.median()]).mean())

# print()

# print(_mean(ser))
# print(_med(ser))
# print(_mix(ser))



# @timer
def benchmark_various_params(model, params, n=None, df=None, 
                             meth=None, save=True) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not n : 
        n = 10

    if      meth == None   : meth = _med
    elif    meth == "mean" : meth = _mean
    elif    meth == "med"  : meth = _med
    elif    meth == "mix"  : meth = _mix
    else                   : raise ValueError("not good method") 

    if save : 
        txt =   "init file         \n"
        txt +=  "model :   {}      \n".format(model)
        txt +=  "params :  {}      \n".format(params)
        txt +=  "n :       {}      \n".format(n)
        txt +=  "meth :    {}      \n".format(meth)

        with open("benchmark_various_params.csv", "w") as f : f.write(txt)

    columns = list(params.keys())
    columns.append("lolo")

    param_dict = combine_param_dict(params)



    param_dict = [
    
                    {'alpha': [0.5623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['tanh'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['invscaling']},
                    {'alpha': [0.5623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['tanh'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['adam'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.1], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [0.01778279410038923], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [3.1622776601683795], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['relu'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [3.1622776601683795], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['tanh'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [0.0001], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.0005623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['invscaling']},
                    {'alpha': [0.0031622776601683794], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [0.1], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['relu'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [17.78279410038923], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['invscaling']},
                    {'alpha': [17.78279410038923], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [0.0005623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.1], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.5623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [3.1622776601683795], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [17.78279410038923], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['identity'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['sgd'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.01778279410038923], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['logistic'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['constant']},
                    {'alpha': [0.1], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['tanh'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['adam'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['adaptive']},
                    {'alpha': [0.5623413251903491], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['relu'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True], 'nesterovs_momentum': [True], 'learning_rate_init': [0.001], 'batch_size': ['auto'], 'max_iter': [200], 'validation_fraction': [0.1], 'epsilon': [1e-08], 'momentum': [0.9], 'warm_start': [True], 'learning_rate': ['invscaling']},
                    {'alpha': [0.0001], 'beta_1': [0.9], 'tol': [0.0001], 'activation': ['logistic'], 'early_stopping': [False], 'hidden_layer_sizes': [(4, 4, 4)], 'solver': ['lbfgs'], 'power_t': [0.5], 'beta_2': [0.999], 'shuffle': [True]}   ]

    results = list()

    for param in param_dict : 

        info("testing param : " + str(param))

        try : 
            lolos = [run_GSCV(model, param, df)[0] for i in range(n)]
            lolo = round(meth(lolos), 3)
            # grid_param = grid.get_params()

            if save : 
                txt = str(lolo) + "," + str(param) + "\n"
                with open("benchmark_various_params.csv", "a") as f : f.write(txt)

            serie = {i: j[0] for i,j in param.items()}
            serie["lolo"] = lolo

            results.append(pd.Series(serie))

            info("done")

        except Exception as e : 

            info("invalid params")
            info(str(param))
            # info(e)

    results = pd.DataFrame(results, columns =columns )
    results.sort_values(by="lolo", ascending=True, inplace=True)

    return results


####

# results = benchmark_various_params(LogisticRegression, all_params, 10, save=True)


# BEST_PARAMS = results.iloc[:10, :]
# BEST_PARAMS


# best_params = results.iloc[0, :]
# best_params


# best_params = {i : [j] for i, j in zip(best_params.index, best_params.values) if i != "lolo"}
# best_params


# # let's try this model n times

# n = 40
# results = [run_GSCV(LogisticRegression, best_params)[0] for i in range(n)]


# # raw results

# pd.Series(results).describe()


# # let's try our theorical best model (feat eng. scoring...)

# _df = standscale(delete_outliers(df,1.3))

# n = 40
# results = [run_GSCV(LogisticRegression, best_params, _df, scoring="roc_auc", cv=10, test_size=0.33)[0] for i in range(n)]



# pd.Series(results).describe()


# # let's definitively find the real best params grid


# params_grid_list = [{i : [j] for i, j in 
#                          zip(BEST_PARAMS.loc[k, :].index, BEST_PARAMS.loc[k, :].values) if i != "lolo"}
#                         for k in BEST_PARAMS.index]

# params_grid_list


# n = 40
# results = [[run_GSCV(LogisticRegression, p)[0] for p in params_grid_list] for i in range(n)]
# results = pd.DataFrame(results, columns = [str("pg_"+str(i)) for i,j in enumerate(params_grid_list)])


# results


# _results = results.describe().T.sort_values(by="mean", axis=0)
# _results


# best_params = params_grid_list[2]
# best_params


# # let's try our theorical best model (feat eng. scoring...)

# _df = standscale(delete_outliers(df,1.3))

# n = 40
# results = [run_GSCV(LogisticRegression, best_params, _df, scoring="roc_auc", cv=10, test_size=0.33)[0] for i in range(n)]


# pd.Series(results).describe()


# #####################################################################################################
# #####################################################################################################

# # 04-making_submission.py

# #####################################################################################################
# #####################################################################################################


# # rferfre
# # ccdsdcds
# # cdcdscdscs
# # cddscdscsdc


# # import 

# # from finding_good_models import * 


p ={ 'penalty': ['l1'],
     'tol': [0.001],
     'class_weight': [None],
     'max_iter': [100],
     'intercept_scaling': [1],
     'multi_class': ['ovr'],
     'solver': ['liblinear'],
     'C': [1],
     'warm_start': [True],
     'dual': [False],
     'fit_intercept': [True]}


params = p


# # warm up

df          = build_df(DATA,  TRAIN_FILE)
df          = delete_outliers(df, 1.4) 

model       = LogisticRegression()

lolo,
     grid  = run_GSCV(       model       = model,
                              params      = params, 
                              df          = df,        
                              cv          = 10, 
                              n_jobs      = 6,    
                              scoring     = "average_precision",
                              verbose     = 1,   
                              test_size   = 0.33)

print(lolo)
print(grid)


# # training

train_df    = build_df(DATA, TRAIN_FILE)
train_df    = delete_outliers(train_df, 1.4)


X,y         = return_X_y(train_df)
grid        = GridSearchCV( estimator   = model, 
                            param_grid  = params,  
                            cv          = 10, 
                            n_jobs      = 6,
                            scoring     = "average_precision", 
                            verbose     = 1)
grid.fit(X,y)


# predicting

test_df     = build_df(DATA, TEST_FILE)
y_pred      = grid.predict_proba(test_df)
y_pred      = y_pred[:,1]
_y_pred     = clipping_log_loss(y_pred, x=0.07)
_y_pred     = np.array(_y_pred)



_y_pred      = pd.Series(y_pred, name="Made Donation in March 2007", index = test_df.index, dtype=np.float64)
path        = finding_master_path("submissions", PROJECT)
path        += "submission0.csv"
_y_pred.to_csv(  path, index=True, header=True, index_label="")

print("done")


# from IPython.display import Image
# Image("../head.png")

