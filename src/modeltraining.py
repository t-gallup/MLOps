import numpy as np
import mlflow
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def neg_rmse(y_true, y_pred):
    return -rmse(y_true, y_pred)

def train_with_tuning(X, y, cv_folds=5, random_seed=42):
    np.random.seed(random_seed)
    rmse_scorer = make_scorer(neg_rmse, greater_is_better=True)
    # Objective function for hyperopt
    def objective(params):
        with mlflow.start_run(nested=True):
            # Initialize model
            regressor_type = params['type']
            model_params = {k: v for k, v in params.items() if k != 'type'}
            if regressor_type == 'lr':
                model = Ridge(**model_params)
            elif regressor_type == 'dt':
                model = DecisionTreeRegressor(**model_params)
            elif regressor_type == 'rf':
                model = RandomForestRegressor(**model_params)
            else:
                return {'loss': float('inf'), 'status': STATUS_OK}
            
            # Perform cross-validation
            cv_scores = cross_val_score(
                model,
                X,
                y,
                cv=cv_folds,
                scoring=rmse_scorer,
                n_jobs=-1
            )
            final_rmse = -cv_scores.mean()

            mlflow.set_tag("Model", regressor_type)
            mlflow.log_params(model_params)
            mlflow.log_metric("RMSE", final_rmse)
            
            return {'loss': final_rmse, 'status': STATUS_OK, 'model_type': regressor_type, 'params': model_params}
    
    # Define hyperparameter search space
    search_space = hp.choice('regressor_type', [
        {
            'type': 'lr',
            'alpha': hp.loguniform('ridge_alpha', np.log(0.001), np.log(100)),
            'tol': hp.loguniform('ridge_tol', np.log(1e-6), np.log(1e-2))
        },
        {
            'type': 'dt',
            'criterion': hp.choice('dtree_criterion', ['squared_error', 'poisson', 'friedman_mse', 'absolute_error']),
            'max_depth': hp.choice('dtree_max_depth', [None, hp.randint('dtree_max_depth_int', 1, 10)]),
            'min_samples_split': hp.randint('dtree_min_samples_split', 2, 10),
            'random_state': random_seed
        },
        {
            'type': 'rf',
            'n_estimators': hp.randint('rf_n_estimators', 20, 500),
            'max_features': hp.randint('rf_max_features', 2, 9),
            'criterion': hp.choice('rf_criterion', ['squared_error', 'poisson', 'friedman_mse', 'absolute_error']),
            'random_state': random_seed
        },
    ])
    
    algo = tpe.suggest
    trials = Trials()
    best_result = fmin(
        fn=objective,
        space=search_space,
        algo=algo,
        max_evals=10,
        trials=trials
    )
    best_trial_idx = np.argmin([trial['result']['loss'] for trial in trials.trials])
    best_trial = trials.trials[best_trial_idx]
    
    # Extract best parameters
    model_type = best_trial['result']['model_type']
    best_params = best_trial['result']['params']
    best_rmse = best_trial['result']['loss']
    
    # Initialize and train the best model
    if model_type == 'lr':
        best_model = Ridge(**best_params)
    elif model_type == 'dt':
        best_model = DecisionTreeRegressor(**best_params)
    elif model_type == 'rf':
        best_model = RandomForestRegressor(**best_params)
    best_model.fit(X, y)

    best_metrics = {
        'model_type': model_type,
        'params': best_params,
        'validation_score': -best_rmse,
        'rmse': best_rmse
    }
    
    print(f"Best model type: {model_type}")
    print(f"Best model RMSE: {best_rmse}")
    
    return best_model, best_metrics
