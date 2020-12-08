'''
Script used to map player based features from avg_metric_features.yml
to team_player_mappings.yml
'''
import lightgbm
import os
import yaml
import pandas as pd
from pathlib import Path
import numpy as np
import sys

#Defines the directories
home_dir = '/Users/Liu/'
gbm_build_dir = os.path.join(home_dir, 'NBA_Pro_Line_Analytics/GBM_model_build/')
config_dir = os.path.join(gbm_build_dir,'feature_generation_scripts/configs/')

scripts_dir = os.path.join(gbm_build_dir, 'model_build_scripts')
sys.path.insert(1, scripts_dir)

import helpers

#Reads in dataset configurations
with open(config_dir + 'dataset_config.yaml', 'r') as stream:
    try:
        dataset_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

gbm_build_dir = Path(gbm_build_dir)
player_config = gbm_build_dir / 'feature_generation_scripts/configs/player_metrics/'

#Opens mapping parameters for the team build dataset
with open(player_config / 'team_player_mappings.yml', 'r') as stream:
    try:
        param_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

data_dir = os.path.join(home_dir, 'NBA_Pro_Line_Analytics/model_build_data/')
data_dir = Path(data_dir)
model_build_team_stats = data_dir / 'NBA_Team_Stats_2010-2019_Rolling_Avg_Win_Loss_pts_diff'
model_build_player_stats = data_dir / 'NBA_Player_Stats_2010-2019_rollling_avg'

team_output_dir = data_dir / 'NBA_Team_Stats_2010-2019_Rolling_Avg_Win_Loss_pts_diff_player_features'

helpers.ensure_dir_exists(team_output_dir)

counter = 0
for dataset in dataset_config['NBA_Team_Stats_2010-2019_Rolling_Avg_Win_Loss_pts_diff']:
    print(f'Now mapping player features for dataset {dataset}')
    team_dataset = pd.read_csv(model_build_team_stats / dataset)
    player_dataset = pd.read_csv(model_build_player_stats / dataset_config['NBA_Player_stats_w_rolling_avg'][counter])
    counter +=1
    for key, value in param_config.items():
        print(f'Now mapping feature {key}')
        team_dataset = helpers.map_player_to_team(team_df = team_dataset,
                                                  player_df = player_dataset,
                                                  param_name = value['Parameter_Name'],
                                                  param_required = value['Feature_required'],
                                                  threshold = value['Floor'])

    data_name = dataset[: -4] + '_player_features.csv'
    team_dataset.to_csv(team_output_dir / data_name, index = False)
