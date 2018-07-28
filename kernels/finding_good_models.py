#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
03_finding_good_models.py
"""


"""
blablabla
blablabla
blablabla
blablabla
"""




# import

from math import ceil
from starting_ML import *


# functions

def combine_param_dict(d) : 

    d = OrderedDict(d)
    combinations = it.product(*(d[feat] for feat in d))
    combinations = list(combinations)

    d = [{i:[j,] for i,j in zip(d.keys(), I)} for I in combinations ]

    return d


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

    best_params_1   = { "C" :[0.11],
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


    if not param :  param = none_params
    else  :         param = best_params_2

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid


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


# @timer
def benchmark_various_params(model, params, n=None, df=None, meth=None, save=True) : 


    if not isinstance(df, pd.DataFrame): 
        df = build_df(DATA, TRAIN_FILE)

    if not n : 
        n = 10

    if      meth == None   : meth = _mix
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



"""
LOGISTIC REGRESSION, NO FEATURES ENG
FOR 10 TESTS :

In [0]: r = benchmark_various_params_once(LogisticRegression, all_params2, 100)

Out[0]:

    {   "solver"            : ["liblinear"],
        "class_weight"      : [None], 
        "dual"              : [False],
        "intercept_scaling" : [1],
        "fit_intercept"     : [True],
        "C"                 : [10],
        "tol"               : [0.001],
        "max_iter"          : [100],
        "warm_start"        : [False],
        "penalty"           : ["l2"],
        "multi_class"       : ["ovr"],  }
    
"""


"""
LOGISTIC REGRESSION, NO FEATURES ENG
FOR 100 TESTS : 

In [0]: r = benchmark_various_params_once(LogisticRegression, all_params2, 100)

Out[0]: 
       solver class_weight   dual  intercept_scaling  fit_intercept     C  \
598        sag         None  False                  1           True  0.10   
138  newton-cg         None  False                  1           True  1.00   
106  newton-cg         None  False                  1           True  0.01   
462  liblinear         None  False                  1           True  1.00   
448  liblinear         None  False                  1           True  1.00   
442  liblinear         None  False                  1           True  0.10   
458  liblinear         None  False                  1           True  1.00   
452  liblinear         None  False                  1           True  1.00   
114  newton-cg         None  False                  1           True  0.10   
282      lbfgs         None  False                  1           True  0.10   

        tol  max_iter  warm_start penalty multi_class   lolo  
598  0.0001       100       False      l2         ovr  0.484  
138  0.0010       100        True      l2         ovr  0.486  
106  0.0010       100        True      l2         ovr  0.486  
462  0.0010       100       False      l2         ovr  0.487  
448  0.0001       100        True      l1         ovr  0.488  
442  0.0010       100        True      l2         ovr  0.488  
458  0.0010       100        True      l2         ovr  0.489  
452  0.0001       100       False      l1         ovr  0.489  
114  0.0001       100        True      l2         ovr  0.489  
282  0.0010       100        True      l2         ovr  0.489  

"""





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

    if not param :  param = none_params
    else  :         param = params

    lolo, grid       = run_GSCV(model, param, None)

    return lolo, grid
