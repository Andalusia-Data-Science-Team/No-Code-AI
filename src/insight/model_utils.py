svc_param_grid = {
    "C": [0.1, 1, 10],
    "kernel": ["linear", "rbf"],
    "gamma": ["scale", "auto"],
}

rf_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}

svr_param_grid = {
    "kernel": ["linear", "rbf"],
    "C": [0.1, 1, 10],
    "epsilon": [0.01, 0.1, 1],
}

lr_param_grid = {"penalty": ["l2"], "C": [0.1, 1, 10]}

lreg_param_grid = {
    "fit_intercept": [True, False],
}

en_param_grid = {
    "alpha": [0.1, 1.0, 10.0],
    "l1_ratio": [0.2, 0.5, 0.8],
    "fit_intercept": [True, False],
}

knn_param_grid = {
    "n_neighbors": [3, 5, 7],
    "weights": ["uniform", "distance"],
    "algorithm": ["auto", "ball_tree", "kd_tree"],
    "leaf_size": [30, 40, 50],
}

XGB_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [3, 5, 7],
    "learning_rate": [0.1, 0.01, 0.001],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

GradientBoosting_param_grid = {
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.1, 0.01, 0.001],
    "max_depth": [3, 5, 7],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "subsample": [0.8, 1.0],
}

ada_param_grid = {"n_estimators": [50, 100, 200], "learning_rate": [0.1, 0.01, 0.001]}

dt_param_grid = {
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

ridge_param_grid = {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]}

lasso_param_grid = {"alpha": [0.1, 1.0, 10.0], "fit_intercept": [True, False]}

xtra_param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10],
    "min_samples_split": [2, 5, 10],
}

grid_dict = {
    "SVC": svc_param_grid,
    "LR": lr_param_grid,
    "Linear Regression": lreg_param_grid,
    "ElasticNet": en_param_grid,
    "KNN_cls": knn_param_grid,
    "KNN_reg": knn_param_grid,
    "RF_cls": rf_param_grid,
    "XGB_cls": XGB_param_grid,
    "XGB_reg": XGB_param_grid,
    "Ridge": ridge_param_grid,
    "Lasso": lasso_param_grid,
    "extra_tree": xtra_param_grid,
    "SVR": svr_param_grid,
    "GradientBoosting_cls": GradientBoosting_param_grid,
    "Adaboost": ada_param_grid,
    "DecisionTree_cls": dt_param_grid,
}
