#!/usr/bin/env python3
#-*- coding: utf8 -*-


"""
03-feature_egineering.py
"""


"""
blablabla
blablabla
blablabla
blablabla
"""



# import

from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, normalize, Normalizer, MinMaxScaler

from improving_naive_models import * 


# functions

def outliers_impact_once(graph=False) : 

    best_params_2   = { "C" :[  0.00001, 0.00003, 0.00006,
                                0.0001, 0.0003, ],
                        "class_weight" :[None], 
                        "dual":[False],
                        "fit_intercept" :[True],
                        "intercept_scaling" :[1],
                        "max_iter" :[100],
                        "multi_class" :["ovr"],
                        "penalty" :["l1"],
                        "solver" :["saga", "liblinear"],
                        "tol":np.logspace(-5, 1, 12),
                        "warm_start" :[True, False]     }

    model = LogisticRegression
    param = best_params_2

    columns = ["k", "lolo", "params"]

    results = pd.DataFrame(columns=columns)

    threshold_list = np.arange(0.9, 5.1, 0.1)
    for i in threshold_list : 


        df = first_tour()
        df = delete_outliers(df, i)

        lolo, grid = run_GSCV(model, param, df)

        r = pd.Series([round(i, 2), lolo, grid.best_params_], index=columns)

        results = results.append(r, ignore_index=True)

    if graph : 
        plt.plot(results.k, results.lolo)
        plt.show()

    return results


def outlier_impacts_multi(n=10, graph=True) : 

    r = outliers_impact_once() 

    results = pd.DataFrame( [outliers_impact_once() for i in range(n)], 
                            columns=r.index)

    if graph : 
         df = pd.DataFrame([results.mean(), results.median()], 
                                            columns=results.mean().index,
                                            index=["mean", "med"])

         df = df.T
         df.plot()
         plt.show()

    return results





def pipe_various_things(): 

    df          = first_tour()
    X,y         = return_X_y(df)
    t           = split(X,y)



    model       = LogisticRegression()
    params      = {}
    cv          = 10
    scoring     = "neg_log_loss"
    verbose     =1
    n_jobs      =5
    grid        = GridSearchCV( estimator=model, 
                                param_grid=params,  
                                cv=cv, 
                                n_jobs=n_jobs,
                                scoring=scoring, 
                                verbose=verbose)


    pipe_grid = Pipeline([      ("scaler", StandardScaler()),
                                ("pca", PCA()),
                                ('model', LogisticRegression()) ])


    X_train, X_test, y_train, y_test = t 
    pipe_grid.fit(X_train, y_train)

    # info(pipe_grid.best_estimator_)
    # info(pipe_grid.best_score_)
    # info(pipe_grid.best_params_)

    y_pred = pipe_grid.predict(X_test)
    lolo = log_loss(y_test, y_pred).round(3)
    info(lolo)


    return lolo, grid






def studying_param_by_param(model, param) : 

    lolo, grid = run_GSCV(model, param)

    gcv = pd.DataFrame(grid.cv_results_)

    p = [i for i in gcv.columns if "param_" in i ][0]
    good = ["mean_test_score", "mean_train_score", "rank_test_score", p]

    gcv = gcv.loc[:, good]

    gcv.set_index(p, drop=True, inplace=True)

    fig, axes = plt.subplots(1, 2)

    gcv1 = gcv.drop("rank_test_score", axis=1)
    gcv1.plot(logx=True, ax=axes[0])

    gcv2 = gcv.rank_test_score
    gcv2.plot(logx=True, ax=axes[1])

    plt.show()





def corr_between_lolo_and_neg_lolo(model, param) :


    k = ["lolo", "neg_lolo"]

    results = pd.DataFrame(columns=k)

    key, values = list(param.items())[0]

    for  val in values : 

        d = {key:[val]}

        lolo, grid = run_GSCV(model, d)

        results = results.append(pd.Series([lolo, grid.best_score_], index=k ), ignore_index=True)


    plt.scatter(results.lolo, results.neg_lolo, marker=".")
    plt.xlabel("lolo")
    plt.ylabel("neg_lolo")
    plt.title("param" + str(key))
    plt.show()

    return results




