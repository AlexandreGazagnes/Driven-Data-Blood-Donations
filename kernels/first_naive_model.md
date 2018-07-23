

```python
#!/usr/bin/env python3
#-*- coding: utf8 -*-


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

```




    '\nin this second part, we will implement our first logistic regression model.\nWe will first implement by hand a naive classifier, then a dummy classifier \n(who does the same job), and finally a basic logistic regression model.\nrather than looking at the results of a regression we will implement a \nfunction that will test the model x times and that will average the results\n obtained\nwe will then implement a results manager that will be a dataframe\n'




```python
# import

# from sklearn.preprocessing import *
from sklearn.model_selection import train_test_split
# from sklearn.grid_search import *

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, log_loss

```


```python
# pasting first_tour.ipynb if nedeed


###############################################################
###############################################################

# -------------------------------------------------------------

# please see first_tour.ipynb before

# -------------------------------------------------------------

###############################################################
###############################################################


# import
import os, sys, logging, random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# logging 
l = logging.WARNING
logging.basicConfig(level=l, format="%(levelname)s : %(message)s")
info = logging.info

# graph
# %matplotlib
# sns.set()

# consts
FOLDER      = "Driven-Data-Blood-Donations"
TRAIN_FILE  = "training_data.csv"
TEST_FILE   = "test_data.csv"

# functions

def finding_master_path(folder="data") :
    path = os.getcwd()
    path = path.split("/")
    idx  = path.index(FOLDER)
    path = path[:idx+1]
    folder = str(folder) + "/"
    path.append(folder)  
    path = "/".join(path)
    if not os.path.isdir(path) : 
        raise NotADirectoryError
    return path
    
def return_datasets(path) : 
    li = [i for i in os.listdir(path) if ".csv" in i ]
    return li 

def build_df(path, file) : 
    df          = pd.read_csv(path+file, index_col=0)
    df.columns  = pd.Index( ["last_don", "num_don","vol_don", "first_don", 
                            "target"], dtype="object")
    return df

def print_df(df) : 
    print(df.ndim)
    print(df.shape)
    print(df.dtypes)
    print(df.index)
    print(df.columns)
    print(df.describe())
    print(df.head(3))
    print(df.tail(3))

def re_dtype(df) : 
    # li = [np.uint8, np.uint16]
    # [print(i,  np.iinfo(i).min, np.iinfo(i).max) for i in li]
    dtypes_dict = {     "last_don"  : np.uint8, 
                        "num_don"   : np.uint8,
                        "vol_don"   : np.uint16, 
                        "first_don" : np.uint8, 
                        "target"    : np.uint8       }
    df = df.astype(dtypes_dict)
    return df 

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
        # sns.distplot --> (kde=True ) ???
        # -----------------------------------------
        axes[i].hist(df[feat], bins=30)
        axes[i].set_title(feat)
    plt.suptitle("features distribution")
    plt.show()

def graph_corr_matrix(df) : 
    corr_mat = df.corr()
    sns.heatmap(corr_mat, cmap="coolwarm", annot=True, fmt='.3g')
    plt.title("correlation matrix")
    plt.show()

def drop_corr_features(df) : 
    df = df.drop("vol_don", axis=1)
    return df 

def study_nas(df) : 
    print(df.isna().any())
    print(df.isna().any())

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

def return_outliers(ser, k) : 
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
    # build data path
    path = finding_master_path("data")
    # info(path)							# UNCOMMENT IF NEEDED
    # just show dataset list
    # datasets = return_datasets(path)      # UNCOMMENT IF NEEDED
    # info(datasets)                        # UNCOMMENT IF NEEDED
    # build our df
    df = build_df(path, file)
    # print main info
    # print_df(df)                          # UNCOMMENT IF NEEDED
    # (overkilled) recast dataframe in a better dtype
    df = re_dtype(df)
    # graph features distr and correlation  # UNCOMMENT IF NEEDED
    # graph_each_feature(df)                  
    # graph_corr_matrix(df)                 # UNCOMMENT IF NEEDED
    # drop corr values
    df = drop_corr_features(df)
    # nas
    # study_nas(df)                         # UNCOMMENT IF NEEDED
    # for i in [1.5, 2, 2.5, 3] :           # UNCOMMENT IF NEEDED
    # study_outliers(df, i)                 # UNCOMMENT IF NEEDED
    # df = delete_outliers(df, 3)           # UNCOMMENT IF NEEDED
    return df

####

df = first_tour()
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>last_don</th>
      <th>num_don</th>
      <th>first_don</th>
      <th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>619</th>
      <td>2</td>
      <td>50</td>
      <td>98</td>
      <td>1</td>
    </tr>
    <tr>
      <th>664</th>
      <td>0</td>
      <td>13</td>
      <td>28</td>
      <td>1</td>
    </tr>
    <tr>
      <th>441</th>
      <td>1</td>
      <td>16</td>
      <td>35</td>
      <td>1</td>
    </tr>
    <tr>
      <th>160</th>
      <td>2</td>
      <td>20</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>358</th>
      <td>1</td>
      <td>24</td>
      <td>77</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
# consts 

COLUMNS = ["naive", "dummy", "basic", "features eng."]
# MODELS = [naive_model, dummy_model, basic_model]

```


```python
# split our features from our target

def return_X_y(df) : 
    
    X = df.drop("target", axis=1)
    y = df.target

    return X, y  

####

X,y = return_X_y(df)

```


```python
# split test and train df/target

def split(X,y) : 

    func = train_test_split
    tup = train_test_split(X, y)

    return tup

####

tup = split(X,y)

```


```python
# build from scratch a naive/dummy model which make prediction regarding global target probabilities

def naive_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    freq = y_test.value_counts() / len(y_test)

    y_pred = np.random.binomial(1, freq[1], len(y_test))
    y_pred = pd.Series(y_pred)

    return accuracy_score(y_test, y_pred).round(3)

####

naive_model(df)

```




    0.646




```python
# rather than conding a dummy model from scratch, use sk learn DummyClassifier (same job) 

def dummy_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    X,y = return_X_y(df)
    t = split(X,y)


    X_train, X_test, y_train, y_test = t 

    model = DummyClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred).round(3)

####

dummy_model()

```




    0.611




```python
# just for fun trying to make predictions with a very basic model (no meta params, no features engineering)
# this one will be our model prediction base
# it is suposed to be better that our DummyClassifier. If not there is a major issue...

def basic_model(df=None) : 

    if not isinstance(df, pd.DataFrame): 
        df = first_tour()

    X,y = return_X_y(df)
    t = split(X,y)

    X_train, X_test, y_train, y_test = t 

    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return accuracy_score(y_test, y_pred).round(3)

####

basic_model()

```




    0.75




```python
# we now need to have a sort of decorator which will be charged to lunch n times our model and to give 
# us the accuracy mean of n trials

def model_accuracy_mean(model, nb=5, df=None) : 

    scores = [model(df) for i in range(nb)]

    info(type(scores))
    info(type(range(nb)))

    score = sum(scores)/len(scores)

    return score.round(3)

####

print(model_accuracy_mean(naive_model))
print(model_accuracy_mean(dummy_model))
print(model_accuracy_mean(basic_model))
```

    0.645
    0.604
    0.747



```python
# we now will be able to build a specific dataframe to handle results of various tested models

COLUMNS = ["naive", "dummy", "basic", "feat eng."]
MODELS =  [naive_model, dummy_model, basic_model]

results = pd.DataFrame(columns=COLUMNS)

####

results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>naive</th>
      <th>dummy</th>
      <th>basic</th>
      <th>feat eng.</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# and for each feature engineering configuration, we will have a function charged to run every 
# listed models and to add properly the results in our specific dataframe

def add_new_results(feat_com=None,
                    results=None, 
                    n=5, 
                    models=MODELS, 
                    columns= COLUMNS,
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

####

results = add_new_results("test")
results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>naive</th>
      <th>dummy</th>
      <th>basic</th>
      <th>feat eng.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.67</td>
      <td>0.603</td>
      <td>0.749</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>




```python
# finally just to test this "meta" model we will test it with our first featue engineering 
# possibility : outilers threshold
# we do not care so much about the results but about the global process of our meta model
# of course if we can find a first way for our feature engineering work, it could be great! 

def first_approch_of_feat_eng(  drop_list,
                                results=None,
                                n=5, models=MODELS, columns=COLUMNS, df=None) : 

    if not isinstance(drop_list, list) : 
        raise TypeError

    if not isinstance(results, pd.DataFrame) : 
        results = pd.DataFrame(columns=columns)

    for i in drop_list : 

        df = first_tour()
        df = delete_outliers(df, i) 

        feat_com = "drop outliers with threshold of k > " + str(i)

        results = add_new_results(  results=results,
                                    feat_com=feat_com,
                                    n=n, 
                                    models=models, 
                                    columns=columns, 
                                    df=df)

    return results

####

results = first_approch_of_feat_eng([1.5, 2, 2.5, 3])
results

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>naive</th>
      <th>dummy</th>
      <th>basic</th>
      <th>feat eng.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.638</td>
      <td>0.630</td>
      <td>0.784</td>
      <td>drop outliers with threshold of k &gt; 1.5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.666</td>
      <td>0.648</td>
      <td>0.781</td>
      <td>drop outliers with threshold of k &gt; 2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.668</td>
      <td>0.637</td>
      <td>0.783</td>
      <td>drop outliers with threshold of k &gt; 2.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.618</td>
      <td>0.643</td>
      <td>0.793</td>
      <td>drop outliers with threshold of k &gt; 3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# just for ther record, let's build a function which resume all this work in 3 ligns

def first_naive_model() : 

    results = pd.DataFrame(columns=COLUMNS)
    results = add_new_results("without_any_feat_eng", results)

    results = first_approch_of_feat_eng([1.5, 2.0, 2.5, 3.0, 3.5], results)

    return results

####

first_naive_model()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>naive</th>
      <th>dummy</th>
      <th>basic</th>
      <th>feat eng.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.608</td>
      <td>0.636</td>
      <td>0.775</td>
      <td>without_any_feat_eng</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.643</td>
      <td>0.653</td>
      <td>0.778</td>
      <td>drop outliers with threshold of k &gt; 1.5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.617</td>
      <td>0.643</td>
      <td>0.760</td>
      <td>drop outliers with threshold of k &gt; 2.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.611</td>
      <td>0.631</td>
      <td>0.777</td>
      <td>drop outliers with threshold of k &gt; 2.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.641</td>
      <td>0.635</td>
      <td>0.772</td>
      <td>drop outliers with threshold of k &gt; 3.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.646</td>
      <td>0.642</td>
      <td>0.771</td>
      <td>drop outliers with threshold of k &gt; 3.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
# conclusion

# We could see through this first study that we are on a dataset quite simple, allowing our first approach 
# to have pretty good results.

# Our base model offers a performance of 0.75 and the impact of the ouliers on the model's performance 
# seems at first glance quite low.

# Having a fairly simple dataset, the possibilities for improving the models will not be that simple due 
# to the small number of variables and the small size of the dataset : the possibilities offered by 
# the feature engineering are indeed quite low.
```
