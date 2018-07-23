#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
02-improving-naive-model.py
"""


"""

"""



# import


from sklearn.model_selection import GridSearchCV



from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import Perceptron, RidgeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC

from first_naive_model import *


# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions



def run_grid(model, params, df=None) : 

	if not isinstance(df, pd.DataFrame): 
		df = first_tour()
	
	X,y = return_X_y(df)
	t = split(X,y)


	cv = 5
	n_jobs=6
	scoring = "accuracy"	# log_loss
	grid = GridSearchCV(	estimator=model, 
							param_grid=params, 	
							cv =cv, 
							n_jobs=n_jobs,
							scoring=scoring, 
							verbose = 1)


	X_train, X_test, y_train, y_test = t 
	grid.fit(X_train, y_train)

	info(grid.best_estimator_)
	info(grid.best_score_)
	info(grid.best_params_)

	y_pred = grid.predict(X_test)
	info(accuracy_score(y_test, y_pred).round(3))

	return grid


def grid_LogisticRegression(df=None) : 
	""" """

	params = {	"penalty" 	  : ["l1", "l2"], #   "solver"	  : [	"newton-cg", "lbfgs", "liblinear", "sag", "saga"],
				"warm_start"  : [True, False], 
				"tol"		  : np.logspace(-6, 2, 9),
				"C" 		  : np.logspace(-4, 2, 7), 
				"max_iter"	  : np.logspace(3, 5, 3) 	} # "multi_class" : ["ovr", "multinomial"]


	model = LogisticRegression()

	grid 	= run_grid(model, params, None)

	return None



def grid_RandomForestClassifier(df=None):
	""" """

	default_params  = 	{	"n_estimators" : [10],
							"criterion" : ["gini"], # or 	"entropy"
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
							"class_weight" : [None] 	}

	none_params 	= {}

	params = 			{	"n_estimators" : [100],
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
							"warm_start" : [True],	}



	model 	= RandomForestClassifier()
	grid 	= run_grid(model, params, None)
	return grid


def grid_AdaBoostClassifier(df=None):
	""" """

	params 	= {}

	model 	= AdaBoostClassifier()
	grid 	= run_grid(model, params, None)
	return grid


def grid_Perceptron(df=None):
	""" """

	params 	= {}

	model 	= Perceptron()
	grid 	= run_grid(model, params, None)
	return grid


def grid_RidgeClassifier(df=None):
	""" """

	params 	= {}

	model 	= RidgeClassifier()
	grid 	= run_grid(model, params, None)

	return None


def grid_KNeighborsClassifier(df=None):
	""" """

	params 	= {}

	model 	= KNeighborsClassifier()
	grid 	= run_grid(model, params, None)

	return None


def grid_MLPClassifier(df=None):
	""" """


	best = 	{	'max_iter':[200], 
			'tol':[2.782559402207126e-06], 
			'solver':['lbfgs'], 
			'activation':['tanh'], 
			'warm_start':[True], 
			'hidden_layer_sizes':[(4, 4, 4)], 
			'alpha':[4.832930238571752]}


	params 	= {	"hidden_layer_sizes": [(4,4,4)],
				"alpha": np.logspace(-5, 1, 20),
				"max_iter": [200, 500],
				"activation": ["logistic", "tanh", "relu"],
				"solver": ["lbfgs", "sgd", "adam"],
				"warm_start": [True],
				"tol": np.logspace(-6, -2, 10)}
				

				# "activation": ["identity", "logistic", "tanh", "relu"],   
				# "solver": ["lbfgs", "sgd", "adam"],   
				# "alpha": np.logspace(-4, 2, 9), 


	params 	= {"hidden_layer_sizes": [(3,3,3), (3,3), (4,4), (5,5), (5,5,5), (4,4,4)],   
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

	model = MLPClassifier()
	grid  = run_grid(model, params, None)

	return None


def grid_LinearSVC(df=None):
	""" """

	params 	= { "penalty":["l2"], # "l1"
				"loss":["hinge","squared_hinge"], 
				"dual":[True], # False
				"tol":np.logspace(-6, 2, 9), 
				"C":np.logspace(-4, 2, 7), 
				"multi_class":["ovr","crammer_singer" ],
				"fit_intercept":[True], 
				"intercept_scaling":[1,], 
				"class_weight":[None,], 
				"max_iter"	: np.logspace(3, 5, 3)		}

	model 	= LinearSVC()
	grid 	= run_grid(model, params, None)

	return None


def grid_NuSVC(df=None):
	""" """

	params 	= {	"nu":[0.5],
				"kernel":["rbf"],
				"degree":[3],
				"gamma":["auto"],
				"coef0":[0.0],
				"shrinking":[True],
				"probability":[False],
				"tol":[0.001],
				"_size":[200],
				"class_weight":[None],
				"max_iter":[-1],
				"decision_function_shape":["ovr"]}

	model 	= NuSVC()

	grid 	= run_grid(model, params, None)

	return None



#############################################################

COLUMNS = [		"naive", 
				"dummy", 
				"basic", 

				"gridLR",
				"gridRF",
				"gridAda"
				"gridPer"
				"gridRC"
				"gridKNN",
				"gridMLP",
				"gridSVC",
				"gridNu",

				"features eng."]

MODELS = [	naive_model, 
			dummy_model, 
			basic_model, 

			grid_LogisticRegression,
			grid_RandomForestClassifier,
			grid_AdaBoostClassifier,
			grid_Perceptron,
			grid_RidgeClassifier,
			grid_KNeighborsClassifier,
			grid_MLPClassifier,
			grid_LinearSVC, 
			grid_NuSVC,]

##############################################################

# results = first_approch_of_feat_eng(results, [1.5, 2.0, 2.5, 3.0, 3.5])
	

