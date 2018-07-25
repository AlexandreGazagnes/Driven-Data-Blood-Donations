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

def caller(funct) : 

    def wrapper(*args, **kwargs) : 

        msg = funct.__name__ + " : called"
        print(msg)
        

        res = funct(*args, **kwargs)

        msg = funct.__name__ + " : ended"
        print(msg)

        return res

    return wrapper


def timer(funct) : 

    def wrapper(*args, **kwargs) : 
        
        t = time.time()

        res = funct(*args, **kwargs)

        t = round(time.time() - t, 2)
        msg = funct.__name__ + " : " + str(t) + " secs" 
        print(msg)

        return res

    return wrapper


# @timer                                    # UNCOMMENT IF NEEDED
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
    

# @timer                                    # UNCOMMENT IF NEEDED
def return_datasets(path) : 

    li = [i for i in os.listdir(path) if ".csv" in i ]
    
    return li 


# @timer                                    # UNCOMMENT IF NEEDED
def build_df(path, file) : 

    df          = pd.read_csv(path+file, index_col=0)

    if len(df.columns) == 5 : 
    	df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            "target"], dtype="object")
    elif len(df.columns) == 4 : 
    	df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            ], dtype="object")
    else : 
        raise ValueError("invalid number")

    return df


# @timer                                    # UNCOMMENT IF NEEDED
def print_df(df) : 

    print(df.ndim)
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    print(df.columns)
    print(df.describe())
    print(df.head(3))
    print(df.tail(3))


# @timer                                    # UNCOMMENT IF NEEDED
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


# @timer                                    # UNCOMMENT IF NEEDED
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


# @timer                                    # UNCOMMENT IF NEEDED
def graph_corr_matrix(df) : 

    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g')

    plt.title("correlation matrix")
    
    plt.show()


# @timer                                    # UNCOMMENT IF NEEDED
def drop_corr_features(df) : 

    df = df.drop("vol_don", axis=1)

    return df 


# @timer                                    # UNCOMMENT IF NEEDED
def study_nas(df) : 

    print(df.isna().any())
    print(df.isna().any())


# @timer                                    # UNCOMMENT IF NEEDED
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


# @timer                                    # UNCOMMENT IF NEEDED
def return_outliers(ser, k) : 

    desc = ser.describe()
    q1, q3, q2 = desc["25%"], desc["75%"], desc["50%"]
    IQ = q3-q1
    range_min, range_max = q1 - k * IQ, q3 + k*IQ

    # outliers = ser[(ser > range_max) or (ser < range_min)]
    
    return ser >= range_max


# @timer                                    # UNCOMMENT IF NEEDED
def delete_outliers(df, k) : 

    li = [i for i in df.columns if "target" not in i]

    for feat in li : 
        df = df[return_outliers(df[feat], k) == False]

    return df

@caller
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





############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################





"""
01-first-naive-model.py
"""


"""
in this second part, we will implement our first logistic regression model.

We will first implement by hand a naive classifier, then a dummy classifier 
(who does the same job), and finally a basic logistic regression model.

rather than looking at the results of a regression we will implement a 
function that will test the model x times and that will average the results
 obtained

we will then implement a results manager that will be a dataframe
"""


# import

# from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import *

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss



# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions

# @timer
def return_X_y(df) : 

    X = df.drop("target", axis=1)
    y = df.target

    return X, y  


# @timer
def split(X,y) : 

    func = train_test_split
    tup = train_test_split(X, y)
    
    return tup


# @timer
def naive_model(df=None) :

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    freq = y_test.value_counts() / len(y_test)
        
    y_pred = np.random.binomial(1, freq[1], len(y_test))
    y_pred = pd.Series(y_pred)

    acc = accuracy_score(y_test, y_pred).round(3)

    return acc, None 


# @timer
def dummy_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()
    
    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred).round(3)

    return acc, model


# @timer
def basic_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()
    
    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred).round(3)
    
    return acc, model


@timer
def model_accuracy_mean(model, nb=5, df=None) : 

    scores = pd.Series([model(df)[0] for i in range(nb)])

    info(type(scores))
    info(type(range(nb)))

    score = scores.mean().round(3)

    return score


##############################################################
##############################################################


COLUMNS = ["naive", "dummy", "basic", "features eng."]
MODELS = [naive_model, dummy_model, basic_model]


##############################################################
##############################################################


# @timer
def add_new_results(results=None, feat_com=None, n=5, 
                    models=MODELS, columns= COLUMNS, 
                    df=None) : 
    
    if not isinstance(results, pd.DataFrame) : 
        results = pd.DataFrame(columns=columns)

    new = [model_accuracy_mean(i, n, df) for i in models]
    info(new)

    if not feat_com : 
        feat_com = "No comment"

    new.append(feat_com)
    info(new)
    
    new = pd.Series(new, index=columns)
    info(new)
    
    results = results.append(new, ignore_index=True)
    info(results)

    return results


@timer
def first_approch_of_feat_eng(  drop_list,
                                results=None,
                                n=5, 
                                models=MODELS, columns= COLUMNS, 
                                df=None) : 
    
    if not isinstance(drop_list, list) : 
        raise TypeError

    if not isinstance(results, pd.DataFrame) : 
        results = pd.DataFrame(columns=columns)

    for i in drop_list : 

        df = first_tour()
        df = delete_outliers(df, i) 

        feat_com = "drop outliers > " + str(i)

        results = add_new_results(  results=results,
                                    feat_com=feat_com,
                                    n=n, 
                                    models=models, 
                                    columns=columns, 
                                    df=df)

    return results


@timer
def first_naive_model() :  

    results = pd.DataFrame(columns=COLUMNS)
    results = add_new_results(results, "without_any_feat_eng")

    results = first_approch_of_feat_eng([1.5, 2.0, 2.5, 3.0, 3.5])
    
    return results









############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################



#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02-improving-naive-model.py
"""


"""
blablabla
blablabla
blablabla
blablabla
"""



# import

import itertools as it
from collections import OrderedDict

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC


# consts 

# COLUMNS   = ["naive", "dummy", "basic", "features eng."]
# MODELS    = [naive_model, dummy_model, basic_model]


# functions


def beeper(funct) : 

    def wrapper(*args, **kwargs) : 
        
        res = funct(*args, **kwargs)

        freq=440
        duration=3
        cmd = 'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq)
        os.system(cmd)

        return res

    return wrapper


# @decor_run_GSCV
# @timer
@caller
def run_GSCV(   model, params, 
                df=None, 
                outliers=None, regularize=None,
                cv=10, n_jobs=6, scoring="accuracy",
                verbose=0) : 

    model = model()

    comment=str()

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    # if outliers : 
    #     try : 
    #         outliers = float(outliers)
    #     except :
    #         raise ValueError("outliers params must be a float")

    #     if not (1.4<= outliers <= 3.5)  :
    #         raise ValueError("outliers must be 1.4<= outliers <= 3.5")

    # if outliers : 
    #     comment+="outliers k= "+ str(outliers)
    #     df = delete_outliers(df, outliers)
     
    X,y = return_X_y(df)

    # if regularize : 
    #     raise ValueError("regularize funct not implemented yet")
    #     X = regularize(X)

    t = split(X,y)

    grid        = GridSearchCV( estimator=model, 
                                param_grid=params,  
                                cv=cv, 
                                n_jobs=n_jobs,
                                scoring=scoring, 
                                verbose=verbose)

    X_train, X_test, y_train, y_test = t 
    grid.fit(X_train, y_train)

    info(grid.best_estimator_)
    info(grid.best_score_)
    info(grid.best_params_)

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred).round(3)
    info(acc)

    return acc, grid


# @decor_grid
# @timer
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
    """
    all_params      = { "penalty":["l1", "l2"],
                        "dual":[True, False],
                        "tol":[0.0001],
                        "C":[1.0],
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "solver":["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                        "max_iter":[100],
                        "multi_class":["ovr", "multinomial"],
                        "warm_start":[False, True],   }
    """

    none_params     = {}

    test_params     = { "penalty":["l2"],
                        "dual":[True],
                        "tol":np.logspace(-6, 2, 9),
                        "C":np.logspace(-4, 2, 7),
                        "fit_intercept":[True],
                        "intercept_scaling":[1],
                        "class_weight":[None],
                        "loss":["hinge","squared_hinge"],
                        "solver":["liblinear"],
                        "max_iter":np.logspace(3, 5, 3),
                        "multi_class":["ovr","crammer_singer" ],
                        "warm_start":[False, True],   }

    best_params_1   = { "C" :[1],
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :[100],
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga"],
                        "tol":[0.0001],
                        "warm_start" :[True]     }

    best_params_2   = { "C" :np.logspace(-4, 2, 7),
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :np.logspace(3, 5, 3),
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga"],
                        "tol":np.logspace(-6, 2, 9),
                        "warm_start" :[True]     }


    if not param :  param = none_params
    else  :         param = best_params_2

    acc, grid       = run_GSCV(model, param, None)

    return acc, grid


# @decor_grid
# @timer
def grid_RidgeClassifier(   df=None, param=None,
                            model=RidgeClassifier):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params


    acc, grid       = run_GSCV(model, param, None)

    return acc, grid


# @decor_grid
# @timer
def grid_LinearSVC(     df=None, param=None,
                        model=LinearSVC):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)

    return acc, grid


# @decor_grid
# @timer
def grid_NuSVC(         df=None,  param=None,
                        model=NuSVC):

    # default_params  = {}

    # none_params     = {}

    # params          = { "nu":[0.5],
    #                     "kernel":["rbf"],
    #                     "degree":[3],
    #                     "gamma":["auto"],
    #                     "coef0":[0.0],
    #                     "shrinking":[True],
    #                     "probability":[False],
    #                     "tol":[0.001],
    #                     "_size":[200],
    #                     "class_weight":[None],
    #                     "max_iter":[-1],
    #                     "decision_function_shape":["ovr"]}

    # if not param :    param = none_params
    # else  :       param = params

    # acc, grid    = run_GSCV(model, param, None)

    # return acc, grid
    return -1.0, None


# @decor_grid
# @timer
def grid_KNeighborsClassifier(  df=None, param=None,
                                model=KNeighborsClassifier):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)

    return acc, grid


# @decor_grid
# @timer
def grid_RandomForestClassifier(df=None, param=None, 
                                model=RandomForestClassifier):

    default_params  = { "n_estimators" : [10],
                        "criterion" : ["gini"], # or    "entropy"
                        "max_features" : ["auto"], # "auto" eq sqrt or"log2", 
                        "max_depth" : [None], 
                        "min_samples_split" : [2],
                        "min_samples_leaf" : [1],
                        "min_weight_fraction_leaf" : [0.0],
                        "max_leaf_nodes" : [None],
                        "min_impurity_decrease" : [0.0],
                        "min_impurity_split" : [None],
                        "bootstrap" : [True],
                        "oob_score" : [False],
                        "n_jobs" : [1],
                        "random_state" : [None],
                        "verbose" : [0],
                        "warm_start" : [False],
                        "class_weight" : [None]     }

    none_params     = {}

    params          = { "n_estimators" : [100],
                        "criterion" : ["gini"],
                        "max_depth" : [None],
                        "min_samples_split" : [2],
                        "min_samples_leaf" : [1],
                        "min_weight_fraction_leaf" : [0.0],
                        "max_features" : [None],
                        "max_leaf_nodes" : [None],
                        "min_impurity_decrease" : [0.0],
                        "min_impurity_split" : [None],
                        "bootstrap" : [True],
                        "oob_score" : [True],
                        "warm_start" : [True],  }

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)
    
    return acc, grid


# @decor_grid
# @timer
def grid_AdaBoostClassifier(df=None, param=None,
                            model=AdaBoostClassifier):

    default_params  = {}

    none_params     = {}

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)
    
    return acc, grid


# @decor_grid
# @timer
def grid_Perceptron(    df= None, param=None,
                        model=Perceptron):

    default_params  = {}

    none_params     = {}

    params          = {}

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)
    
    return acc, grid


# @decor_grid
# @timer
def grid_MLPClassifier( df=None, param=None,
                        model=MLPClassifier):

    default_params  = {}

    none_params     = {}

    params          = { "hidden_layer_sizes": [(4,4,4)],
                        "alpha": np.logspace(-5, 1, 20),
                        "max_iter": [200, 500],
                        "activation": ["logistic", "tanh", "relu"],
                        "solver": ["lbfgs", "sgd", "adam"],
                        "warm_start": [True],
                        "tol": np.logspace(-6, -2, 10)}
        
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

    if not param :  param = none_params
    else  :         param = params

    acc, grid       = run_GSCV(model, param, None)

    return acc, grid


#############################################################
#############################################################


COLUMNS = [     "dummy", 
                "basic", 

                "gridLR",
                "gridRC",

                "gridSVC",
                "gridNu",

                "gridKNN",

                "gridRF",

                "gridAda", 

                "gridPer",
                "gridMLP",

                "features eng."]

MODELS = [      dummy_model, 
                basic_model, 

                grid_LogisticRegression,
                grid_RidgeClassifier,

                grid_LinearSVC, 
                grid_NuSVC,

                grid_KNeighborsClassifier,

                grid_RandomForestClassifier,

                grid_AdaBoostClassifier,

                grid_Perceptron,
                grid_MLPClassifier]


#############################################################
##############################################################


@timer
def benchmark_various_grid_once(col, mod, n=10) : 

    results     = [model_accuracy_mean(m, n, None) for m in MODELS]
    results     = pd.Series(results, index=col[:-1])

    info(results)

    return results


@timer
def benchmark_various_grid_multi(col, mod, N=10, n=10) : 

    benchmark   = benchmark_various_grid_once
    series      = [benchmark(col, mod, n) for i in range(N)]
    results     = pd.DataFrame(series, columns=col[:-1])

    info(results)

    info(results.describe())
    info((results.T).describe())

    return results


def combine_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


@timer
def benchmark_various_params_once(model, params, n) : 

    param_dict = combine_dict(params)

    results = list()

    columns = list(params.keys())
    columns.append("acc")

    for param in param_dict : 

        info("testing param : " + str(param))

        try : 
            accs = [run_GSCV(model, param)[0] for i in range(n)]
            accs = np.array(accs)
            acc = accs.mean().round(3)
            # grid_param = grid.get_params()

            info("done")

        except Exception as e : 

            acc, grid_param = -100.0, "invalid dict of args"
            info("invalid params")
            info(str(param))
            info(e)

        serie = {i: j[0] for i,j in param.items()}
        serie["acc"] = acc

        results.append(pd.Series(serie))

    results = pd.DataFrame(results, columns =columns.append("acc") )

    # clean nas in acc
    mask = results.acc != -100.0
    results = results[mask]

    results.sort_values(by="acc", ascending=False, inplace=True)

    return results


def graph_benchmark_various_params_once(model, params, n_list=[1, 3, 5,]) : # 10, 20, 50

    color_list = ["red", "green", "blue", "grey", "black", "yellow"]

    meta_results = list()

    for n, c in zip(n_list, color_list) : 
        results = benchmark_various_params_once(model, params, n)

        m, s = results.acc.mean(), results.acc.std()

        plt.scatter(results.index, results.acc, c=c, marker=".")
        plt.plot(results.index, [m for i in results.index ], c=c)
        plt.plot(results.index, [m+s for i in results.index ], c=c,linestyle="dashed")
        plt.plot(results.index, [m-s for i in results.index ], c=c, linestyle="dashed")
        txt = "n={}, m={}, s={}".format(n,m,s)
        # plt.label(txt)

        meta_results.append((n, results))

    plt.title("result mean, std and val regarding numb of exp")
    plt.show()

    return meta_results




def main() : 

    df = first_tour("data", TRAIN_FILE)
    df = delete_outliers(df, 3.3)






    best_params_2   = { "C" :[0.0001],
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :[100],
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga", "liblinear"],
                        "tol":np.logspace(-4, 1, 5),
                        "warm_start" :[True, False]     }


    model = LogisticRegression
    param = best_params_2

    acc, grid = run_GSCV(model, param, df)

    print("acc = {}".format(acc))
    input()


    path = finding_master_path("data")
    df = build_df(path, TEST_FILE)
    df = drop_corr_features(df)

    y = grid.predict_proba(df)
    y = y[:,0]

    y = pd.Series(y, name="Made Donation in March 2007", index = df.index, dtype=np.float64)
    
    path = finding_master_path("submissions")
    path += "submission2.csv"
    y.to_csv(   path, index=True, 
                header=True, index_label="")



# if __name__ == '__main__':
#   main()








