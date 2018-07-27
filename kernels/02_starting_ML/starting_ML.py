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

    def wrapper(*args, **kwargs) : 
        
        res = funct(*args, **kwargs)

        freq=440
        duration=3
        cmd = 'play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq)
        os.system(cmd)

        return res

    return wrapper


# @timer
# @caller
def run_GSCV(   model=None, params=None, 
                df=None,
                cv=0, 
                n_jobs=None, 
                scoring=None,
                verbose=None, 
                test_size=None) : 


    if not model : 
        model = LogisticRegression()
    else : 
        model = model()

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not params : 
        params = dict()

    if not cv : 
        cv=5

    if not n_jobs : 
        n_jobs = 6

    if not scoring : 
        scoring = "accuracy"

    if not verbose : 
        verbose=1

    if not test_size : 
        test_size = 0.33
     
    X,y         = return_X_y(df)
        
    X_train, X_test, y_train, y_test = split(X,y, test_size)

    grid = None

    info("\n\nbefore grid init")
    try :       
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
        print(e)
        input()

    info("\n\nbefore fit")
    try : 
        grid.fit(X_train, y_train)
        info("grid fit OK")
        info(grid.best_estimator_)
        info(grid.best_score_)
        info(grid.best_params_)

    except Exception as e : 
        info("grit fit went wrong")
        print(e)
        input()


    info("\n\nbefore pred")
    try :
        y_pred = grid.predict_proba(X_test)
        y_pred = y_pred[:,1]
        info("pred OK")
        info("run_GSCV 0") 
    
    except Exception as e : 
        info("pred went wrong")
        print(e)
        input()

        info("maybe predict_proba do not exists just predict")
        try : 
            y_pred = grid.predict(X_test)
            info("second pred Method OK")
            info("run_GSCV 1")

        except Exception as e : 
            info("2nd pred method went wrong")
            print(e)
            input()

    info("\n\nbefore computing lolo")
    try : 

        lolo = log_loss(y_test, y_pred).round(3)
        info("lolo ok")
        info(lolo)

    except Exception as e : 
        info("lolo went wrong")
        print(e)
        input()


    if isinstance(lolo, float) and grid : 

        return lolo, grid

    raise ValueError("run_GSCV error")

#############################################################
#############################################################


COLUMNS = [     "dummy", "basic",
                "LR",       "RC",
                "SVC",      # "Nu",
                "KNN",
                "DT", 
                "RF", 
                "Ada", 
                "Per",      "MLP",

                "features eng."]

MODELS = [      dummy_model, basic_model, 
                LogisticRegression, RidgeClassifier,
                LinearSVC, # NuSVC,
                KNeighborsClassifier,
                DecisionTreeClassifier,
                RandomForestClassifier,
                AdaBoostClassifier,
                Perceptron, MLPClassifier]


#############################################################
##############################################################


# @timer
def benchmark_various_models_once(  n=5, df=None, 
                                    models = MODELS, columns= COLUMNS) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    models  = models
    columns = columns[:-1]
    
    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    results = list()

    for m in models : 
        r = [run_GSCV(m, dict(), df) for i in range(n)]
        r = pd.Series([i[0] for i in r])
        r = r.mean().round(3)
        results.append(r)

    results     = pd.Series(results, index=columns)

    info(results)

    results.sort_values(ascending=True, inplace=True)

    return results


# @timer
def benchmark_various_models_multi(   N=10, n=10, df=None, graph=True,
                                    models = MODELS, columns= COLUMNS) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    models  = models
    columns = columns[:-1]
    
    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    benchmark   = benchmark_various_models_once
    series      = [benchmark(n, df) for i in range(N)]
    results     = pd.DataFrame(series, columns=columns)

    info(results)

    info(results.describe())
    info((results.T).describe())

    results = results.describe()
    results = results.T
    results.sort_values(by="50%", ascending=True, inplace=True)
    results = results.T

    if graph : 
        fig, ax = plt.subplots(1,2, figsize=(13,13))
        ax = ax.flatten()

        results.boxplot(ax=ax[0])
        _results = results.iloc[:,:5]
        _results.boxplot(ax=ax[1])
        
        plt.show()

    return results



def benchmark_various_outliers_once(n=5, df=None, k_list=None, model=None ) :

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not isinstance(k_list, Iterable) : 
        k_list = np.arange(10,50,1)
        k_list = (k_list/10).round(1) 

    if not model : 
        model = LogisticRegression

    results = list()

    for k in k_list :

        _df = delete_outliers(df, k) 

        if not isinstance(_df , pd.DataFrame) : 
            raise TypeError("we need a pd.DataFrame here")

        r = [run_GSCV(model, dict(), _df) for i in range(n)]
        r = pd.Series([i[0] for i in r])
        r = r.mean().round(3)
        results.append(r)

    results     = pd.Series(results, index=k_list)

    info(results)


    return results



##############################################################
##############################################################


COLUMNS = [     "basic",
                "LR",       "RC",
                "SVC",      # "Nu",
                "KNN",
                "DT", 
                "RF", 
                "Ada", 
                "Per",      "MLP",

                "features eng."]

MODELS = [      basic_model, 
                LogisticRegression, RidgeClassifier,
                LinearSVC, # NuSVC,
                KNeighborsClassifier,
                DecisionTreeClassifier,
                RandomForestClassifier,
                AdaBoostClassifier,
                Perceptron, MLPClassifier]


##############################################################
##############################################################


def benchmark_various_outliers_multi(   N=10, n=10, df=None, graph=True,
                                        model = None, k_list=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not isinstance(k_list, Iterable) : 
        k_list = np.arange(10,50,1)
        k_list = (k_list/10).round(1) 

    if not model : 
        model = LogisticRegression


    benchmark   = benchmark_various_outliers_once
    series      = [benchmark(n, df, k_list, model) for i in range(N)]
    results     = pd.DataFrame(series, columns=k_list)

    info(results)

    info(results.describe())
    info((results.T).describe())

    results = results.describe()
    results = results.T
    results.sort_values(by="50%", ascending=True, inplace=True)
    results = results.T

    if graph : 
        fig, ax = plt.subplots(1,1, figsize=(13,13))
        ax = ax.flatten()

        results.boxplot(ax=ax[0])
        
        plt.show()

    return results


def benchmark_various_outliers_models(n=5, df=None, graph=True,
                                    k_list=None, models = MODELS, columns= COLUMNS) : 
    
    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not isinstance(k_list, Iterable) : 
        k_list = np.arange(10,50,1)
        k_list = (k_list/10).round(1) 

    models  = models
    columns = columns[:-1]
    
    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    for model, name in zip(models, columns) : 
        ser = benchmark_various_outliers_once(n, df, k_list, model)
        ser.name = name

        ser.plot()

    plt.legend()
    plt.ylabel("log_loss")
    plt.xlabel("outlier threshold 'k'")
    plt.show()


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


###############################################################
###############################################################


TRANSFORM_LIST = [nothing, normalize, standscale, min_max]
TRANSFORM_INDEX = ["nothing", "normalize", "standscale", "min_max"]


###############################################################
###############################################################


def benchmark_various_transform_once(   n=10, df=None, model=None,
                                        transform_list=TRANSFORM_LIST, transform_index = TRANSFORM_INDEX) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not transform_list : 
        transform_list = [nothing, normalize, standscale, min_max]
 
    if not transform_index : 
        transform_index = ["nothing", "normalize", "standscale", "min_max"]

    if not model : 
        model =  LogisticRegression

    results = list()
    for transform in transform_list : 

        _df = transform(df) 

        if not isinstance(_df , pd.DataFrame) : 
            raise TypeError("we need a pd.DataFrame here")

        r = [run_GSCV(model, dict(), _df) for i in range(n)]
        r = pd.Series([i[0] for i in r])
        r = r.mean().round(3)
        results.append(r)

    results     = pd.Series(results, index=transform_index)

    info(results)

    return results


def benchmark_various_transform_multi(  N=5, n=5, df=None, graph = True, model=None,
                                        transform_list=TRANSFORM_LIST, transform_index = TRANSFORM_INDEX) : 

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not transform_list : 
        transform_list = [nothing, normalize, standscale, min_max]
 
    if not transform_index : 
        transform_index = ["nothing", "normalize", "standscale", "min_max"]

    if not model : 
        model = LogisticRegression


    benchmark   = benchmark_various_transform_once
    series      = [benchmark(n, df, model, transform_list, transform_index) for i in range(N)]
    results     = pd.DataFrame(series, columns=transform_index)

    info(results)

    info(results.describe())
    info((results.T).describe())

    results = results.describe()
    results = results.T
    results.sort_values(by="50%", ascending=True, inplace=True)
    results = results.T

    if graph : 
        fig, ax = plt.subplots(1,1, figsize=(13,13))

        results.boxplot(ax=ax)
        
        plt.show()

    return results


def benchmark_various_transform_models(  n=10, df=None, graph=True, 
                                        transform_list=None, models = MODELS, columns= COLUMNS) : 
    
    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    models  = MODELS
    columns = COLUMNS[:-1]
    
    if len(models) != len(columns) : 
        raise ValueError("lens not goods")

    for model, name in zip(models, columns) : 
        ser = benchmark_various_transform_once(n, df, None, model)
        ser.name = name

        ser.plot()

    plt.show()



# test_size = [0.4, 0.33, 0.25]
# scoring = ["accuracy", "neg_loss_loss"]
    


def benchmark_various_grid_once(n=5, df=None, model=None, param_type="cv") : 

    cv_list = [int(i) for i in np.arange(2,11, 1)]
    test_size_list =  [round(float(i), 2) for i in (np.arange(20, 41, 1) /100)]
    scoring_list = ['accuracy', 'neg_log_loss', 'f1', 'average_precision', 'precision', 'recall']

    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if param_type == "cv" : 
        test_size_list  = None
        scoring_list    = None

    elif param_type == "test_size" : 
        cv_list         = None
        scoring_list    = None
    
    elif param_type = "scoring": 
        cv_list         = None
        test_size_list  = None

    if not model : 
        model =  LogisticRegression

    results = list()

    if cv_list : 
        for cv in cv_list : 

            r = [run_GSCV(model, dict(), df, cv=cv) for i in range(n)]
            print(r)
            r = pd.Series([i[0] for i in r])
            r = r.mean().round(3)
            results.append(r)

        results     = pd.Series(results, index=cv_list)

    elif test_size_list : 

        for test_size in test_size_list : 

            r = [run_GSCV(model, dict(), df, test_size=test_size) for i in range(n)]
            print(r)
            r = pd.Series([i[0] for i in r])
            r = r.mean().round(3)
            results.append(r)

        results     = pd.Series(results, index=test_size_list)


    elif scoring_list : 
        results = [[run_GSCV(scoring=s)[0] for s in score_list] for i in range(n)]
        results = pd.DataFrame([pd.Series(i, index=score_list) for i in results], columns=score_list)



        
    info(results)

    return results




    
# def benchmark_various_grid_multi(df=None) : 
    
#     pass


def combine_param_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


