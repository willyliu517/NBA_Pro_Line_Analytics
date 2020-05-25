import os
import pandas as pd
from sklearn.model_selection import train_test_split
from model_build_scripts import helpers
import lightgbm
import yaml
import pickle


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
        
        
        
                 
                                             
                                           
        
        

   