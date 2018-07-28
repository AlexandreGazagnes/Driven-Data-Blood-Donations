#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
zjidadod
nudazadai
dzuadiaaudai
"""


# import 

from finding_good_models import * 

# grid params


p = {   "solver"            : ["liblinear"],
        "class_weight"      : [None], 
        "dual"              : [False],
        "intercept_scaling" : [1],
        "fit_intercept"     : [True],
        "C"                 : [10],
        "tol"               : [0.00001],
        "max_iter"          : [100],
        "warm_start"        : [False],
        "penalty"           : ["l2"],
        "multi_class"       : ["ovr"],  }

"""
p = {   "solver"            : ["sag"],
        "class_weight"      : [None], 
        "dual"              : [False],
        "intercept_scaling" : [1],
        "fit_intercept"     : [True],
        "C"                 : [0.1],
        "tol"               : [0.0001],
        "max_iter"          : [100],
        "warm_start"        : [False],
        "penalty"           : ["l2"],
        "multi_class"       : ["ovr"],  }
"""

p = {   "solver"            : "liblinear",
        "class_weight"      : None, 
        "dual"              : False,
        "intercept_scaling" : 1,
        "fit_intercept"     : True,
        "C"                 : 10,
        "tol"               : 0.00001,
        "max_iter"          : 100,
        "warm_start"        : False,
        "penalty"           : "l2",
        "multi_class"       : "ovr",  }


params = p

model = LogisticRegression(**params)
model = AdaBoostClassifier(model)


p = {   "n_estimators"      : [50, 100, 300, 600, 1000, 1200, 1300],
        "learning_rate"     : np.logspace(-4,2, 7)}


params = p


# # warm up
# df          = build_df(DATA, TRAIN_FILE)
# df          = standscale(df)

# lolo, grid  = run_GSCV(     model       = model,     
#                             params      = params, 
#                             df          = df,        
#                             cv          = 10, 
#                             n_jobs      = 6,    
#                             scoring     = "average_precision",
#                             verbose     = 1,   
#                             test_size   = 0.33, 
#                             debug_mode  = False)

# print(lolo)
# print(grid)
# input()


# training

train_df    = build_df(DATA, TRAIN_FILE)
train_df    = standscale(train_df)

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
test_df     = standscale(test_df)

y_pred      = grid.predict_proba(test_df)
y_pred      = y_pred[:,1]


# saving

y_pred      = pd.Series(y_pred, name="Made Donation in March 2007", index = test_df.index, dtype=np.float64)
path        = finding_master_path("submissions", PROJECT)
path       += "submission4.csv"
y_pred.to_csv(  path, index=True, header=True, index_label="")

print("done")





