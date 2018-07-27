#!/usr/bin/env python3
#-*- coding: utf8 -*-


from finding_best_model import * 



params 		= 	{	"tol":[0.0001, 0.001, ],                   # consider also np.logspace(-6, 2, 9)
            		"C":  [0.0001,0.001, 0.01, 0.1, 1]	}     # consider also np.logspace(-3, 1, 40)

train_df	= build_df(DATA, TRAIN_FILE)

X,y 		= return_X_y(train_df)

grid        = GridSearchCV( estimator=LogisticRegression, 
                            param_grid=params,  
                            cv=10, 
                            n_jobs=6,
                            scoring="accuracy", 
                            verbose=1)

grid.fit(X,y)

test_df		= build_df(DATA, TEST_FILE)

y_pred 		= grid.predict_proba(test_df)
y_pred 		= y_pred[:,1]

y_pred 		= pd.Series(y_pred, name="Made Donation in March 2007", index = test_df.index, dtype=np.float64)
    
path 		= finding_master_path("submissions", PROJECT)
path 	   += "submission4.csv"

y_pred.to_csv(  path, index=True, 
                header=True, index_label="")