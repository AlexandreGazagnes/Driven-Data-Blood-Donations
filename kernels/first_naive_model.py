#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
01-first-naive-model.py
"""



"""

"""


# import

# from sklearn.grid_search import *
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
# from sklearn.preprocessing import *
from sklearn.linear_model import LogisticRegression

from first_tour import *


# functions


def return_X_y(df) : 
	""" """

	X = df.drop("target", axis=1)
	y = df.target

	return X, y  


def naive_model(y) : 
	""" """

	freq = y.value_counts() / len(y)
		
	pred = np.random.binomial(1, freq[1], len(y))
	pred = pd.Series(pred)

	return accuracy_score(y, pred)


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

	return accuracy_score(y_test, y_pred)


def basic_model(X_train, X_test, y_train, y_test) : 
	""" """

	model = LogisticRegression()
	model.fit(X_train, y_train)
	y_pred = model.predict(X_test)

	return accuracy_score(y_test, y_pred)


# df = first_tour()
# X,y = return_X_y(df)
# t = split(X,y)

# print(naive_model(y))

# print(basic_model(*t))