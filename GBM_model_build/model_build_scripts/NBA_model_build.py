import os
import pandas as pd
from sklearn.model_selection import train_test_split
from model_build_scripts import helpers
import lightgbm
import yaml
import pickle
import numpy as np
from hyperopt.pyll.base import scope 
from hyperopt import fmin, STATUS_OK, Trials, hp, tpe, rand
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class NBA_Model_Build:
    
    #initializes the class
    def __init__(self , scenario_name, feature_yaml, scenario_yaml):
        self.scenario_name = scenario_name
        self.feature_yaml = feature_yaml
        self.scenario_yaml = scenario_yaml
        self.parse_config()
        
    def parse_config(self):
        "Method of reading in the scenario configurations"
        feature_config = helpers.load_yaml_file(self.feature_yaml)
        scenario_config = helpers.load_yaml_file(self.scenario_yaml)
        
        self.feature_config = feature_config
        self.scenario_config = scenario_config
        
        self.usable_filters = self.feature_config['usable_filters']
        self.usable_var_set = self.feature_config['usable_feature_set']
        
        #grabs the scenario needed from the scenario yaml config file
        self.scenario = [scenario for scenario in scenario_config if scenario['scenario_name'] == self.scenario_name][0]
        scenario_param = self.scenario['scenario_param']
        
        self.input_path = os.path.expanduser(scenario_param['inputs_path'])
        self.results_path = helpers.ensure_dir_exists(os.path.expanduser(scenario_param['results_path']))
        self.seed = scenario_param['seed']
        self.target = scenario_param['target']
        self.candidate_var_set = scenario_param['candidate_variable_set']
        self.X_vars = list()
        [self.X_vars.extend(self.usable_var_set[var_set]) for var_set in self.candidate_var_set]
        self.X_vars = list(set(self.X_vars))
        self.model_build_filter = self.usable_filters['date_filter_build']
        self.model_validation_filter = self.usable_filters['date_filter_validation']
        self.fmin_max_evals = scenario_param['hyperopt']['fmin_max_evals']
        self.hyperopt_param_space = helpers.eval_space(scenario_param['hyperopt']['hyperparam_space'])
                                                       
                                                       
    def load_data(self):
        ''' Method for reading in the model build data'''
        df = pd.read_csv(self.input_path)
        ''' Applies filter used for build and validation dataset'''
        df_build = df[eval(self.model_build_filter)]
        df_train, df_tune = train_test_split(df_build, test_size = .2, random_state = self.seed)
        df_validate = df[eval(self.model_validation_filter)]
        self.df = df
        self.df_train = df_train
        self.df_tune = df_tune
        self.df_validate = df_validate
        
    def run_rfe(self, model_params, target, X_vars, threshold = 0, model_type = 'indicator'):
        if model_type == 'indicator':
            model = lightgbm.LGBMModel(**model_params, importance_type = 'gain')
        elif model_type == 'regressor':
            model = lightgbm.LGBMRegressor(**model_params, importance_type = 'gain')
        eval_set = [(self.df_tune[X_vars], self.df_tune[self.target])]
        model.fit(X =self.df_train[X_vars],
                  y = self.df_train[self.target],
                  eval_set = eval_set,
                  verbose = False)
        
        #Dataframe of features and their corresponding level of importance by gain
        importance_df = pd.DataFrame(data = {
                                              'features': X_vars,
                                              'gain_importances': model.feature_importances_ 
                                             })
        
        while sum(model.feature_importances_ <= threshold) > 0:
            print(f"{sum(model.feature_importances_ <= threshold)} features below threshold")
            print("The following features will be removed:")
            print(importance_df.loc[model.feature_importances_ <= threshold]['features'].tolist())
            
            
            features = importance_df.loc[model.feature_importances_ > threshold]['features'].tolist()
            eval_set = [(self.df_tune[features], self.df_tune[self.target])]
            model.fit(X =self.df_train[features],
                      y = self.df_train[self.target],
                      eval_set = eval_set,
                      verbose = False)
            importance_df = pd.DataFrame(data = {
                                               'features': model.booster_.feature_name(),
                                               'gain_importances': model.feature_importances_ 
                                                })
            
        self.feature_importance_df = importance_df.sort_values(by=['gain_importances'], ascending=False)
        self.post_rfe_model = model
        return self.post_rfe_model, self.feature_importance_df
    
    def run_hyperopt(self, param_space, X_vars, model_params, fmin_max_evals,
                     algo = 'tpe', metric = 'balanced_accuracy_score', 
                     trials_obj = None, model_type = 'indicator'): 
        '''
        Function to run Bayeisan or Random Search hyperparameter optimization
        '''
        
        #Builds the model object to conduct hyperparameter tuning on 
        if model_type == 'indicator':
            hyperopt_model = lightgbm.LGBMModel(**model_params, importance_type = 'gain')
        elif model_type == 'regressor':
            hyperopt_model = lightgbm.LGBMRegressor(**model_params, importance_type = 'gain')
        eval_set = [(self.df_tune[X_vars], self.df_tune[self.target])]
        hyperopt_model.fit(X =self.df_train[X_vars],
                           y = self.df_train[self.target],
                           eval_set = eval_set,
                           verbose = False)
        data = self.df_tune
        
        def evaluate_metric(params):
            
            hyperopt_model.set_params(**params, bagging_freq  = 1 ).fit(X =self.df_train[X_vars],
                                                                        y = self.df_train[self.target],
                                                                        eval_set = eval_set,
                                                                        verbose = False)
            
            eval_x = data[X_vars]
            y_true = data[self.target]
            
            y_score = hyperopt_model.predict(eval_x)
            
            y_pred =  [np.argmax(i) for i in y_score]
            
            if isinstance(metric, str):
                sk_scorer = getattr(metrics, metric, None)
            if sk_scorer is None:
                print(f"Specified metric {metric} does not exist in sklearn")
            
            score = sk_scorer(y_true = y_true, y_pred = y_pred)
            
            return {'loss': -score, 'params': params, 'status': STATUS_OK }
        
        if trials_obj is None:
            self.trials = Trials()
        else:
            self.trials = trials_obj
            
        if algo == 'tpe':
            algo = tpe.suggest 
        elif algo == 'random':
            algo = rand.suggest
            
        best_params = fmin(
            evaluate_metric,
            space = param_space,
            algo = algo,
            max_evals = fmin_max_evals,
            rstate = np.random.RandomState(self.seed),
            trials = self.trials
        )
        
        return best_params, self.trials
            
        
 