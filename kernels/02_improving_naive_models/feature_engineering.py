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

from improving_naive_models import * 


# functions

def outliers_impact_once(graph=False) : 

    best_params_1   = { "C" :[1],
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

    model = LogisticRegression
    param = best_params_1

    results = list()

    threshold_list = np.arange(1.4, 3.7, 0.1)
    for i in threshold_list : 


        df = first_tour()
        df = delete_outliers(df, i)

        lolo = run_GSCV(model, param, df)[0]

        results.append([round(i, 2), lolo])

    if graph : 
        x, y = zip(*results)
        plt.plot(x, y)
        plt.show()


    results = pd.Series([i[1] for i in results], index = [i[0] for i in results])

    return results


def outlier_impacts_multi(n=10, graph=False) : 

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

