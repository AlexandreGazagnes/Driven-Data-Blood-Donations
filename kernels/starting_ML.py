#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02_starting_ML.py
"""


"""
blablabla
blablabla
blablabla
blablabla
"""



# import

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

from first_naive_models import *


# consts 

# COLUMNS   = ["naive", "dummy", "basic", "features eng."]
# MODELS    = [naive_model, dummy_model, basic_model]


# functions

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

# @timer
# @caller
def run_GSCV(   model=None,     params=None, 
                df=None,        tup=None,
                cv=0,           n_jobs=None,
                scoring=None,   test_size=None, 
                verbose=None,   debug_mode=None) : 

    # init default params
    if not model        : model = LogisticRegression()
    else : 
        try             : model = model()
        except          : pass

    is_df               = isinstance(df, pd.DataFrame)
    if not is_df        : df = build_df(DATA, TRAIN_FILE)
    if not params       : params = dict()
    if not cv           : cv=5
    if not n_jobs       : n_jobs = 6
    if not scoring      : scoring = "accuracy"
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
        lolo = log_loss(y_te, y_pred).round(3)
        info("lolo ok")
        info(lolo)

    except Exception as e : 
        info("lolo went wrong")
        if debug_mode : input()
        raise(e)
    

#############################################################
#############################################################


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

    if graph :  
        results.boxplot()
        plt.xlabel("models")
        plt.ylabel("log_loss score")
        plt.title("benchmark various models, without feat eng or meta params")
        plt.show()

    results = results.describe()
    results = results.T.sort_values(by="50%").T

    return results


def benchmark_various_outliers(     n=5, df=None, graph=True, params=None,
                                    k_list=None, model=None ) :

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not isinstance(k_list, Iterable) : 
        k_list = np.arange(10,50,1)
        k_list = (k_list/10).round(1) 

    if not model : model = LogisticRegression

    if not params : params = dict() 

    results = [ pd.Series([run_GSCV(model, params, delete_outliers(df, k))[0] for k in k_list], 
                        index=k_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=k_list)

    if graph : 
        results.boxplot()
        # pd.Series[df.mean().round(2), index=k_list].plot()
        plt.xlabel("outliers 'k' values")
        plt.ylabel("log_loss score")
        plt.title("benchmark various outliers 'k' values, without feat eng or meta params")
        plt.show()

    return results.describe()


# def benchmark_various_outliers_models(n=5, df=None, graph=True, params=None,
#                                     k_list=None, models = MODELS, columns= COLUMNS) : 

    
    # if not isinstance(df, pd.DataFrame): 
    #     df = build_df(DATA, TRAIN_FILE)

    # if not isinstance(k_list, Iterable) : 
    #     k_list = np.arange(10,50,1)
    #     k_list = (k_list/10).round(1) 

    # if len(models) != len(columns) : 
    #     raise ValueError("lens not goods")

    # results = [     pd.Series([run_GSCV(m, params, df)[0] for m in models], 
    #                     index=columns) for i in range(n)]
    
    # results = pd.DataFrame(results, columns=columns)

    # if graph :  
    #     results.boxplot()
    #     plt.xlabel("models")
    #     plt.ylabel("log_loss score")
    #     plt.title("benchmark various models, without feat eng or meta params")
    #     plt.show()

    # results = results.describe()
    # results = results.T.sort_values(by="50%").T

    # return results


    # for model, name in zip(models, columns) : 
    #     ser = benchmark_various_outliers_once(n, df, k_list, model)
    #     ser.name = name

    #     ser.plot()

    # plt.legend()
    # plt.ylabel("log_loss")
    # plt.xlabel("outlier threshold 'k'")
    # plt.show()


#############################################################
#############################################################


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
    

def min_max(df=None) : 

    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    return transform_df(MinMaxScaler, df)


TRANSFORM_LIST = [nothing, normalize, standscale, min_max]
TRANSFORM_INDEX = ["nothing", "normalize", "standscale", "min_max"]


def benchmark_various_transform(   n=10, df=None, graph=True, params=None, model=None,
                                    transform_list=TRANSFORM_LIST, transform_index = TRANSFORM_INDEX) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not model : model =  LogisticRegression

    if not params : params = dict() 

    if len(transform_list) != len(transform_index) : 
        raise ValueError("lens not goods")

    results = [     pd.Series([run_GSCV(model, params, transf(df))[0] for transf in transform_list], 
                        index=transform_index) for i in range(n)]
    
    results = pd.DataFrame(results, columns=transform_index)

    if graph :  
        results.boxplot()
        plt.xlabel("transformations of df")
        plt.ylabel("log_loss score")
        plt.title("benchmark various df transforms, without feat eng or meta params")
        plt.show()

    results = results.describe()
    results = results.T.sort_values(by="50%").T

    return results


# def benchmark_various_transform_models(  n=10, df=None, graph=True, params=None, 
#                                         transform_list=None, models = MODELS, columns= COLUMNS) : 
    
#     if not isinstance(df, pd.DataFrame): 
#         df = build_df(DATA, TRAIN_FILE)

#     if len(models) != len(columns) : 
#         raise ValueError("lens not goods")

#     for model, name in zip(models, columns) : 
#         ser = benchmark_various_transform_once(n, df, None, model)
#         ser.name = name

#         ser.plot()

#     plt.show()


#############################################################
#############################################################


def benchmark_various_scoring(  n=5, df=None, graph=True, params=None, model=None,
                                scoring_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not scoring_list : 
        scoring_list = ['accuracy', 'neg_log_loss', 'f1', 'average_precision', 'precision', 'recall']

    if not model : model = LogisticRegression

    if not params : params = dict() 

    results = [ pd.Series([run_GSCV(model, params, df, scoring=s)[0] for s in scoring_list], 
                        index=scoring_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=scoring_list)

    if graph :  
        results.boxplot()
        plt.xlabel("scoring methods for grid search")
        plt.ylabel("log_loss score")
        plt.title("benchmark various scoring, without feat eng or meta params")
        plt.show()

    results = results.describe()
    results = results.T.sort_values(by="50%").T

    return results


def benchmark_various_cv(   n=5, df=None, graph=True, params=None, model=None,
                            cv_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not cv_list : 
        cv_list = [int(i) for i in np.arange(2,11, 1)]

    if not model : 
        model = LogisticRegression

    if not params : params = dict() 

    results = [ pd.Series([run_GSCV(model, params, df, cv=c)[0] for c in cv_list], 
                        index=cv_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=cv_list)

    if graph :  
        results.boxplot()
        plt.xlabel(" nb of kfolds for grid search")
        plt.ylabel("log_loss score")
        plt.title("benchmark various nb of kfolds, without feat eng or meta params")
        plt.show()

    results = results.describe()

    return results


def benchmark_various_test_size(   n=5, df=None, graph=True, params=None, model=None,
                                   test_size_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not test_size_list : 
        test_size_list =  [round(float(i), 2) for i in (np.arange(20, 41, 1) /100)]

    if not model : 
        model = LogisticRegression

    if not params : params = dict() 

    results = [ pd.Series([run_GSCV(model, params, df, test_size=t)[0] for t in test_size_list], 
                        index=test_size_list) for i in range(n)]
    
    results = pd.DataFrame(results, columns=test_size_list)

    if graph :  
        results.boxplot()
        plt.xlabel("nb of test_size for test/train split")
        plt.ylabel("log_loss score")
        plt.title("benchmark various test_size for test/train split, without feat eng or meta params")
        plt.show()

    results = results.describe()

    return results


# def combine_param_dict(d) : 
#
#    d = OrderedDict(d)
#    combinations = it.product(*(d[feat] for feat in d))
#    combinations = list(combinations)
#
#    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]
#
#    return d


def threshold_log_loss(y_pred, y_test=None, x=0.05) : 
    
    if not isinstance(y_pred, Iterable) : 
        raise ValueError("y_pred has to be a pd.Series")
        
    if isinstance(y_test, pd.Series) :  
        info(log_loss(y_test, y_pred))
    
    if not(0.00999 <= x <= 0.4999 ) :
        raise ValueError("threshold must be 0.01 --> 0.5")
 
    y_pred = pd.Series(y_pred)
    y_pred = y_pred.apply(lambda i : x if i<= x else i)
    
    x = round((1-x), 2)
    y_pred = y_pred.apply(lambda i : x if i>= x else i)

    if isinstance(y_test, pd.Series) :  
        info(log_loss(y_test, y_pred))
    
    return y_pred


def threshold_lolo_gain(k, df, model=None, params=None) : 
    
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

    y_pred = threshold_log_loss(y_pred, k)
    new_lolo = log_loss(y_test, y_pred)

    return round((init_lolo - new_lolo) / init_lolo,3)


def benchmark_various_threshold_lolo(   n=20, df=None, graph=True, params=None, 
                                        model=None, threshold_list=None) : 
    
    if not isinstance(df, pd.DataFrame) : 
        df = build_df(DATA, TRAIN_FILE)

    if not model : 
        model = LogisticRegression()
        
    if not params : 
        params = dict()

    if not threshold_list : 
        threshold_list = [round(i/100, 2) for i in range(1,50)]
        # threshold_list = [round(i/1000, 3) for i in range(10,101)]
    
    results = [ [threshold_lolo_gain(k, df) for k in threshold_list]
                     for _ in range(n)]
    
    results = pd.DataFrame(results, columns=threshold_list)
    
    return results


