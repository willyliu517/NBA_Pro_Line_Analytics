'''
    Module where all helper functions are written
'''
import os
import yaml
import pandas as pd
from hyperopt import hp
from hyperopt.pyll.base import scope

def load_yaml_file(path):
    "loads in the yaml file specified in the path"
    with open(path, 'r') as stream:
        try:
            config = yaml.load(stream)
            return config
        except yaml.YAMLError as exc:
            print(exc)


def ensure_dir_exists(dir_path):
    "checks if the directory exists and creates it if not"
    try:
        os.makedirs(dir_path)
        return dir_path
    except FileExistsError:
        return dir_path

def eval_space(hyperparam_space):
    '''
        Need to evaluate hyperparameter space since they are python code and not strings
    '''
    for key in hyperparam_space.keys():
        hyperparam_space[key] = eval(hyperparam_space[key])
    return hyperparam_space

def map_player_to_team(team_df, player_df, param_name, param_required, threshold):
    #Declares the Road Team and Home Team variables
    param_ht = param_name + '_HT'
    param_rt = param_name + '_RT'
    team_df[param_ht] = ''
    team_df[param_rt] = ''

    #Defines list of game_id to iterate over
    list_game_id = team_df['GAME-ID']

    team_df = team_df.set_index('GAME-ID')

    for game_id in list_game_id:
        team_df_temp = team_df[team_df.index == game_id]
        player_df_temp = player_df[player_df['GAME-ID'] == game_id]
        #Identifies the home and road team for each GAME-ID
        home_team = team_df_temp['TEAM_HT'].iloc[0]
        away_team = team_df_temp['TEAM_RT'].iloc[0]
        ht_player_df_team = player_df_temp[player_df_temp.player_team == home_team]
        rt_player_df_team = player_df_temp[player_df_temp.player_team == away_team]
        #Maps player data only if the number of Null values for required column is less than 5
        if sum(ht_player_df_team[param_required].isnull()) < 5:
            team_df.loc[game_id,  param_ht] = int(sum(ht_player_df_team[param_required] > threshold))
        if sum(rt_player_df_team[param_required].isnull()) < 5:
            team_df.loc[game_id, param_rt] = int(sum(rt_player_df_team[param_required] > threshold))

    return team_df.reset_index()
