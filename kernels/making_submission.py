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


# p = {   "solver"            : ["liblinear"],
#         "class_weight"      : [None], 
#         "dual"              : [False],
#         "intercept_scaling" : [1],
#         "fit_intercept"     : [True],
#         "C"                 : [10],
#         "tol"               : [0.00001],
#         "max_iter"          : [100],
#         "warm_start"        : [False],
#         "penalty"           : ["l2"],
#         "multi_class"       : ["ovr"],  }


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

# p = {   "solver"            : "liblinear",
#         "class_weight"      : None, 
#         "dual"              : False,
#         "intercept_scaling" : 1,
#         "fit_intercept"     : True,
#         "C"                 : 10,
#         "tol"               : 0.00001,
#         "max_iter"          : 100,
#         "warm_start"        : False,
#         "penalty"           : "l2",
#         "multi_class"       : "ovr",  }


params = p

# model = LogisticRegression(**params)
# model = AdaBoostClassifier(model)

model = LogisticRegression()


# p = {   "n_estimators"      : [50, 100, 300, 600, 1000, 1200, 1300],
#         "learning_rate"     : np.logspace(-4,2, 7)}


# params = p


# warm up
df          = build_df(DATA, TRAIN_FILE)
df          = standscale(df)

X,y = return_X_y(df)
X_tr, X_te, y_tr, y_te = split(X, y)
y_test = y_te

grid = GridSearchCV(model, params, 
                        cv = 10, 
                        n_jobs=6,
                        scoring="average_precision")
    
grid.fit(X_tr, y_tr)
    
y_pred = grid.predict_proba(X_te)
y_pred = y_pred[:, 1]

init_lolo = lolo = log_loss(y_test, y_pred)
print(init_lolo)
y_pred = threshold_log_loss(y_pred, 0.05)
new_lolo = log_loss(y_test, y_pred)
print(new_lolo)
print(round((init_lolo - new_lolo) / init_lolo,3))

input()


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
y_pred      = threshold_log_loss(y_pred, 0.05)

# saving

y_pred      = pd.Series(y_pred, name="Made Donation in March 2007", index = test_df.index, dtype=np.float64)
path        = finding_master_path("submissions", PROJECT)
path       += "submission5.csv"
y_pred.to_csv(  path, index=True, header=True, index_label="")

print("done")





