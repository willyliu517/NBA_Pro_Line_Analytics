'''
Script used to compute features that are outcome related
'''
import os
import pandas as pd
import numpy as np
import yaml
import sys

#Defines the directories
home_dir = '/Users/Liu/'
gbm_build_dir = os.path.join(home_dir, 'NBA_Pro_Line_Analytics/GBM_model_build/')
config_dir = os.path.join(gbm_build_dir,'feature_generation_scripts/configs/')

scripts_dir = os.path.join(gbm_build_dir, 'model_build_scripts')
sys.path.insert(1, scripts_dir)

import helpers

with open(config_dir + 'dataset_config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#Function used to classify basketball game outcomes as defined by PROLINES guidelines
# 0 - Away team wins by 6+ points
# 1 - Game is within 5 points
# 2 - Home team wins by 6+ points
#score diff is calculated as Home Team Score - Away Team Score
def outcome_maker(df):
    if abs(df['score_diff']) <= 5:
        return 1
    elif df['score_diff'] >= 6:
        return 2
    elif df['score_diff'] <= -6:
        return 0

#Function used to return the winning team from the match
def winning_team(df):
    if df['score_diff'] < 0:
        return df['TEAM_RT']
    else:
        return df['TEAM_HT']

#Function used to return the losing team from the match
def losing_team(df):
    if df['score_diff'] < 0:
        return df['TEAM_HT']
    else:
        return df['TEAM_RT']


def team_won(df):
    if df['winning_team'] == team:
        return 1
    else:
        return 0

def team_lost(df):
    if df['losing_team'] == team:
        return 1
    else:
        return 0

def score_diff(df):
    if df[team + '_won'] == 0:
        return -1 * abs(df['SCORE_DIFF'])
    else:
        return abs(df['SCORE_DIFF'])

output_dir = helpers.ensure_dir_exists(os.path.join(home_dir, 'NBA_Pro_Line_Analytics', 'model_build_data/NBA_Team_Stats_2010-2019_Rolling_Avg_Win_Loss/'))
rolling_avg_dir = os.path.join(home_dir, 'NBA_Pro_Line_Analytics', 'model_build_data/NBA_Team_Stats_2010-2019_rolling_avg/')

for data in config['NBA_Team_Stats_2010-2019_rolling_avg']:
    df_new = pd.read_csv(os.path.join(rolling_avg_dir, data))
    #Separates the df out by Home Team and Away Team
    home_teams = df_new[df_new.VENUE == 'H']
    road_teams = df_new[df_new.VENUE == 'R']
    #Joins the two sets by HT and RT suffixes
    match_df = home_teams.merge(road_teams, how='left', on=['GAME-ID',
                                                            'DATE',
                                                            'DATASET'], suffixes=('_HT', '_RT'))
    #Score diff calculated as Home Team PTS minus Away Team PTS
    match_df['score_diff'] = match_df['PTS_HT'] - match_df['PTS_RT']
    #Outputs the wining and losing team of the match
    match_df['winning_team'] = match_df.apply(winning_team, axis = 1)
    match_df['losing_team'] = match_df.apply(losing_team, axis = 1)
    #Creates target variables for model (used in multi-classification)
    match_df['outcome'] = match_df.apply(outcome_maker, axis=1)
    # Binary indicator for outcome of the game within 5
    match_df['outcome_within_5'] = match_df['score_diff'].apply(lambda x: 1 if abs(x) <= 5 else 0)
    # Binary indicator for home team winning by 6+
    match_df['outcome_ht_6_plus'] = match_df['score_diff'].apply(lambda x: 1 if x >= 6 else 0)
    # Binary indicator for road team winning by 6+
    match_df['outcome_rt_6_plus'] = match_df['score_diff'].apply(lambda x: 1 if x <= -6 else 0)
    # Binary indicator for home team winning by 11+
    match_df['outcome_ht_11_plus'] = match_df['score_diff'].apply(lambda x: 1 if x >= 11 else 0)
    # Binary indicator for road team winning by 11+
    match_df['outcome_rt_11_plus'] = match_df['score_diff'].apply(lambda x: 1 if x <= -11 else 0)
    # Returns the winning and losing teams from the match
    match_df['winning_team'] = match_df.apply(winning_team, axis=1)
    match_df['losing_team'] = match_df.apply(losing_team, axis=1)

    for team in match_df.TEAM_HT.unique():
        match_df[team + '_won'] = match_df.apply(team_won, axis=1)
        match_df[team + '_lost'] = match_df.apply(team_lost, axis=1)

    with open(os.path.join(config_dir, 'team_metrics/num_wins_losses_params.yml'), 'r') as stream:
        try:
            win_loss_param_config = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for namekey, value in win_loss_param_config.items():
        print(f'Now computing feature {namekey}')
        match_df['HT_' + namekey] = ''
        match_df['RT_' + namekey] = ''
        for team in match_df.TEAM_HT.unique():
            if value['column_required'] == 'team_won':
                aa = match_df[(match_df.TEAM_HT == team) |
                              (match_df.TEAM_RT == team)][team + '_won'].rolling(
                              value['rolling_window_required']).sum().shift().to_frame()
                key = team + '_won'
            elif value['column_required'] == 'team_lost':
                aa = match_df[(match_df.TEAM_HT == team) |
                              (match_df.TEAM_RT == team)][team + '_lost'].rolling(
                              value['rolling_window_required']).sum().shift().to_frame()
                key = team + '_lost'

            aa_ht = aa.loc[np.where(match_df.TEAM_HT == team)]
            aa_rt = aa.loc[np.where(match_df.TEAM_RT == team)]

            aa_ht = aa_ht.rename(columns={key: 'HT_' + namekey})
            match_df.update(aa_ht)
            aa_rt = aa_rt.rename(columns={key: 'RT_' + namekey})
            match_df.update(aa_rt)

    data_name = data[0:len(data) - 5] + '_win_loss.csv'
    match_df.to_csv(os.path.join(output_dir, data_name), index = False)

print(f'Finished computing win-loss feature metrics.')
