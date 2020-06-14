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

class NBA_Model_Build:
    
    #initializes the class
    def __init__(self , scenario_name, feature_yaml = './configs/features_config.yml', scenario_yaml = './configs/scenarios_config.yml'):
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
        df_train, df_tune = train_test_split(df_build, test_size = .25, random_state = self.seed)
        df_validate = df[eval(self.model_validation_filter)]
        self.df = df
        self.df_train = df_train
        self.df_tune = df_tune
        self.df_validate = df_validate
        
    def run_rfe(self, model_params, target, X_vars, threshold = 0):
        model = lightgbm.LGBMModel(**model_params, importance_type = 'gain')
        eval_set = [(self.df_tune[X_vars], self.df_tune[self.target])]
        model.fit(X =self.df_train[X_vars],
                  y = self.df_train[self.target],
                  eval_set = eval_set)
        
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
                      eval_set = eval_set)
            importance_df = pd.DataFrame(data = {
                                               'features': model.booster_.feature_name(),
                                               'gain_importances': model.feature_importances_ 
                                                })
            
        self.feature_importance_df = importance_df.sort_values(by=['gain_importances'], ascending=False)
        self.post_rfe_model = model
        return self.post_rfe_model, self.feature_importance_df
    
    def run_hyperopt(self, param_space, X_vars, model_params, fmin_max_evals,
                     algo = 'tpe', metric = 'balanced_accuracy_score', trials_obj = None): 
        '''
        Function to run Bayeisan or Random Search hyperparameter optimization
        '''
        
        #Builds the model object to conduct hyperparameter tuning on 
        hyperopt_model = lightgbm.LGBMModel(**model_params, importance_type = 'gain')
        eval_set = [(self.df_tune[X_vars], self.df_tune[self.target])]
        hyperopt_model.fit(X =self.df_train[X_vars],
                           y = self.df_train[self.target],
                           eval_set = eval_set)
        data = self.df_tune
        
        def evaluate_metric(params):
            
            hyperopt_model.set_params(**params, bagging_freq  = 1 ).fit(X =self.df_train[X_vars],
                                                                        y = self.df_train[self.target],
                                                                        eval_set = eval_set)
            
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
            
        
    def run_feature_reduction(self, model_params, Xvars, metric = 'balanced_accuracy_score', reduction_step = .15):
        '''
        Method used to assess the number of features to include in the final model
        
        '''
        
        selected_list = Xvars
        
        num_of_features = len(selected_list)
        
        sk_scorer = getattr(metrics, metric, None)
        
        score_list = []
        num_features_list = []
        
        feature_selection_df = pd.DataFrame()
        idx = 0
        while num_of_features > 20: 
            
            num_features_list.append(num_of_features)
            
            print(f"building out model with {num_of_features} features:")
            model = lightgbm.LGBMModel(**model_params, importance_type = 'gain')
            tuning_set = [(self.df_tune[selected_list], self.df_tune[self.target])]
            model.fit(X = self.df_train[selected_list], y = self.df_train[self.target], eval_set = tuning_set)
            
            tune_x = self.df_tune[selected_list]
            y_true = self.df_tune[self.target]
            y_score = model.predict(tune_x)
            y_pred = [np.argmax(i) for i in y_score]
            
            score = sk_scorer(y_true = y_true, y_pred = y_pred)
            score_list.append(score)
            
            #Creates the feature importance dataframe
            importance_df = pd.DataFrame(data = {
                                               'features': model.booster_.feature_name(),
                                               'gain_importances': model.feature_importances_ 
                                                })
        
            importance_df = importance_df.sort_values(by=['gain_importances'], ascending=False)
            
            list_of_features = importance_df['features']
            
            feature_selection_df.insert(loc = idx, column = f'{num_of_features}_features', value = list_of_features)
            idx += 1
            
            num_of_features = int(len(list_of_features) * ( 1 - reduction_step))
            
            selected_list = list_of_features[0: num_of_features]
            
        with PdfPages(feature_plot_path) as pdf:
            fig = plt.figure(figsize = (20,10))
            ax = fig.add_subplot(111)
            ax.plot(num_features_list, score_list)
            plt.ylabel('balanced_accuracy_score')
            plt.xlabel('num_of_features')
            pdf.savefig(fig)
            
        return feature_selection_df
            
        
            
                
            
            
                
            
                
    
           
        
            
            
            
            
        
        
        
        
        
                 
                                             
                                           
        
        

   