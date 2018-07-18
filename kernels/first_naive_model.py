#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
01-first-naive-model.py
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

# from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import *

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from first_tour import *



# consts 

# COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]


# functions


def return_X_y(df) : 
	""" """

	X = df.drop("target", axis=1)
	y = df.target

	return X, y  


def naive_model(X_train, X_test, y_train, y_test) : 
	""" """

	freq = y_test.value_counts() / len(y_test)
		
	y_pred = np.random.binomial(1, freq[1], len(y_test))
	y_pred = pd.Series(y_pred)

	return accuracy_score(y_test, y_pred).round(3)


def split(X,y) : 
	""" """

	func = train_test_split
	tup = train_test_split(X, y)
	
	return tup


def dummy_model(X_train, X_test, y_train, y_test) : 
	""" """

	model = DummyClassifier()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return accuracy_score(y_test, y_pred).round(3)


def basic_model(X_train, X_test, y_train, y_test) : 
	""" """

	model = LogisticRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return accuracy_score(y_test, y_pred).round(3)


def model_accuracy_mean(model, nb, X_train, X_test, y_train, y_test) : 
	""" """

	scores = [model(X_train, X_test, y_train, y_test) for i in range(nb)]

	score = sum(scores)/len(scores)

	return score.round(3)



#############################################################

COLUMNS = ["naive", "dummy", "basic", "features eng."]
MODELS = [naive_model, dummy_model, basic_model]

##############################################################




def add_new_results(results, feat_com, t, n=100, models=MODELS, columns= COLUMNS) : 
	""" """
	
	new = [model_accuracy_mean(i, n, *t) for i in models]
	info(new)

	new.append(feat_com)
	info(new)
	
	new = pd.Series(new, index=columns)
	info(new)
	
	results = results.append(new, ignore_index=True)
	info(results)

	return results


def first_approch_of_feat_eng(results, drop_list) : 
	""" """
	
	if not isinstance(drop_list, list) : 
		raise TypeError

	for i in drop_list : 
		df = first_tour()
		df = delete_outliers(df, i) 
		X,y = return_X_y(df)
		t = split(X,y)
		comment = "drop outliers > " + str(i)
		results = add_new_results(results, comment,t)

	return results


def first_naive_model() : 
	""" """	
	df = first_tour()
	X,y = return_X_y(df)
	t = split(X,y)

	results = pd.DataFrame(columns=COLUMNS)
	results = add_new_results(results, "without_any_feat_eng", t)

	results = first_approch_of_feat_eng(results, [1.5, 2.0, 2.5, 3.0, 3.5])
	
	return results



# results = first_naive_model()


