#!/usr/bin/env python3
#-*- coding: utf8 -*-




# import

import os, sys, logging, random

import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import seaborn as sns

# from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV


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
    

def return_datasets(path) : 
    """ """

    li = [i for i in os.listdir(path) if ".csv" in i ]
    
    return li 


def build_df(path, file) : 
    """ """


    df          = pd.read_csv(path+file, index_col=0)
    df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            "target"], dtype="object")


    return df


def print_df(df) : 
    """ """

    print(df.ndim)
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    print(df.columns)
    print(df.describe())
    print(df.head(3))
    print(df.tail(3))



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



def graph_each_feature(df)  : 
    """ """

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


def graph_corr_matrix(df) : 
    """ """

    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g')

    plt.title("correlation matrix")
    
    plt.show()


def drop_corr_features(df) : 
    """ """

    df = df.drop("vol_don", axis=1)

    return df 


def study_nas(df) : 
    """ """

    print(df.isna().any())
    print(df.isna().any())


def study_outliers(df, k=1.5) : 
    """ """

    
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


def return_outliers(ser, k) : 
    """ """

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

def first_tour(folder="data", file=TRAIN_FILE) : 


    # build data path
    path = finding_master_path(folder)
    info(path)

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


    # df = delete_outliers(df, 3)            # UNCOMMENT IF NEEDED


    return df




def return_X_y(df) : 
	""" """

	X = df.drop("target", axis=1)
	y = df.target

	return X, y  


def naive_model(df=None) : 
	""" """
	if not isinstance(df, pd.DataFrame): 
		df = first_tour()

	X,y = return_X_y(df)
	t = split(X,y)

	X_train, X_test, y_train, y_test = t 

	freq = y_test.value_counts() / len(y_test)
		
	y_pred = np.random.binomial(1, freq[1], len(y_test))
	y_pred = pd.Series(y_pred)

	return accuracy_score(y_test, y_pred).round(3)


def split(X,y) : 
	""" """

	func = train_test_split
	tup = train_test_split(X, y)
	
	return tup


def dummy_model(df=None) : 
	""" """

	if not isinstance(df, pd.DataFrame): 
		df = first_tour()
	
	X,y = return_X_y(df)
	t = split(X,y)


	X_train, X_test, y_train, y_test = t 

	model = DummyClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return accuracy_score(y_test, y_pred).round(3)


def basic_model(df=None) : 
	""" """

	if not isinstance(df, pd.DataFrame): 
		df = first_tour()
	
	X,y = return_X_y(df)
	t = split(X,y)


	X_train, X_test, y_train, y_test = t 

	model = LogisticRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return accuracy_score(y_test, y_pred).round(3)


def model_accuracy_mean(model, nb=5, df=None) : 
	""" """

	scores = [model(df) for i in range(nb)]

	info(type(scores))
	info(type(range(nb)))

	score = sum(scores)/len(scores)

	return score.round(3)



def grid_search_logistic_regression(df=None) : 
	""" """


	if not isinstance(df, pd.DataFrame): 
		df = first_tour()
	
	X,y = return_X_y(df)
	t = split(X,y)


	params = {	"penalty" 	  : ["l1", "l2"], #   "solver"	  : [	"newton-cg", "lbfgs", "liblinear", "sag", "saga"],
				"warm_start"  : [True, False], 
				"tol"		  : np.logspace(-6, 2, 9),
				"C" 		  : np.logspace(-4, 2, 7), 
				"max_iter"	  : np.logspace(2, 5, 4) 	} # "multi_class" : ["ovr", "multinomial"]

	model = LogisticRegression()
	cv = 10
	n_jobs=3
	scoring = "accuracy"		# log_loss
	grid = GridSearchCV(	estimator=model, 
							param_grid=params, 	
							cv =cv, 
							n_jobs=n_jobs,
							scoring=scoring )
	
	X_train, X_test, y_train, y_test = t 
	grid.fit(X_train, y_train)

	info(grid.best_estimator_)
	info(grid.best_score_)
	info(grid.best_params_)

	y_pred = grid.predict(X_test)

	return accuracy_score(y_test, y_pred).round(3)




def main() : 

	df = first_tour("data", TRAIN_FILE)

	grid = grid_search_logistic_regression(df)


	path = finding_master_path("data")
	df = build_df(path, TEST_FILE)
	df = drop_corr_features(df)

	y = grid.predict(df)
	y = pd.Series(y, name="Made Donation in March 2007", index = df.index, dtype=np.floap64)
	
	# path = finding_master_path("submissions")
	# path += "submission0.csv"
	# y.to_csv(	path, index=True, 
	# 			header=True, index_label="")


# if __name__ == '__main__':
# 	main()