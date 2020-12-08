import pandas as pd
import yaml
import os
import sys
from pathlib import Path

home_dir = Path('/Users/Liu/')

gbm_build_dir = home_dir / 'NBA_Pro_Line_Analytics/GBM_model_build/'
config_dir = gbm_build_dir / 'feature_generation_scripts/configs/'

with open(config_dir / 'dataset_config.yaml', 'r') as stream:
    try:
        dataset_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

#loads in parameter config containing averge metrics
with open(config_dir / 'player_metrics' / 'avg_metric_features.yml' , 'r') as stream:
    try:
        player_metric_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

scripts_dir = os.path.join(gbm_build_dir, 'model_build_scripts')
sys.path.insert(1, scripts_dir)

import helpers

#Function used to create indicators for whether a player recorded double digit stats for any of the categories
#listed in stats_cols
def double_digit_ind(df, stat_cols = ['PTS', 'A', 'TOT', 'BL', 'ST'] ):
    '''
    Function used to generate indicators for whether the player has recorded double
    '''
    cols_needed = []
    for stat in stat_cols:
        col_name = stat + '_doub_digit_ind'
        cols_needed.append(col_name)
        df[col_name] = df[stat].apply(lambda x: 1 if x >= 10 else 0)

    df['num_cats_doub_digit'] = df[cols_needed].sum(axis = 1)

player_dir = home_dir / 'NBA_Pro_Line_Analytics/raw_data/NBA_Player_Stats_2010-2019/'

output_dir = helpers.ensure_dir_exists(home_dir / 'NBA_Pro_Line_Analytics/model_build_data/NBA_Player_Stats_2010-2019_rollling_avg/')

i = 0

for data in dataset_config['NBA_Player_stats_raw']:
    print(f'Now reading in datset {data}')
    player_stats = pd.read_excel(player_dir / data, sheet_name=0)

    print(f'Renaming features:')
    player_stats = player_stats.rename(columns = {'PLAYER \nFULL NAME': 'player_name',
                                                  'OWN \nTEAM': 'player_team',
                                                  'OPPONENT \nTEAM': 'opposing_team',
                                                  'VENUE\n(R/H)': 'venue',
                                                  'STARTER\n(Y/N)': 'starter_ind',
                                                  'USAGE \nRATE (%)': 'usage_rate',
                                                  'DAYS\nREST': 'days_rested',
                                                  'PLAYER-ID' : 'player_id'
                                                  })

    double_digit_ind(player_stats)
    #Indicators used to generate whether a player has recorded a triple double or double double
    player_stats['triple_double_ind'] = player_stats['num_cats_doub_digit'].apply(lambda x: 1 if x == 3 else 0)
    player_stats['double_double_ind'] = player_stats['num_cats_doub_digit'].apply(lambda x: 1 if x == 2 else 0)
    player_stats['efficiency'] =  player_stats['PTS'] + player_stats['TOT'] + player_stats['ST'] + player_stats['BL'] - player_stats['TO'] - (player_stats['FGA'] - player_stats['FG'])
    player_stats['TS_pct'] = player_stats['PTS'] / (2 * (player_stats['FGA'] + .44 * player_stats['FTA']) )
    # Computes the rolling mean from the data
    for key, value in player_metric_config.items():
        print(f'Now computing {key} for {data}')
        player_stats = player_stats.set_index(['GAME-ID'])
        poww = pd.DataFrame()
        for player in player_stats.player_id.unique():
            temp_df = player_stats[player_stats.player_id == player]
            if value['method_of_compute'] == 'mean':
                aa = temp_df.groupby('player_id')[value['column_required']].rolling(value['rolling_window_required']).mean().shift().reset_index()
                key_name = value['column_required']
            elif value['method_of_compute'] == 'sum':
                aa = temp_df.groupby('player_id')[value['column_required']].rolling(value['rolling_window_required']).sum().shift().reset_index()
                key_name = value['column_required']
            elif value['method_of_compute'] == 'pct_calc':
                aa = temp_df.groupby('player_id')[value['column_required']].rolling(value['rolling_window_required']).sum().shift().reset_index()
                aa[value['Parameter_Name']] = aa[value['column_required'][0]] / aa[value['column_required'][1]]
                aa = aa.drop(columns = value['column_required'], axis = 1)
                key_name = value['Parameter_Name']
            elif value['method_of_compute'] == 'eff_pct_calc':
                aa = temp_df.groupby('player_id')[value['column_required']].rolling(value['rolling_window_required']).sum().shift().reset_index()
                aa[value['Parameter_Name']] = (aa[value['column_required'][0]] + .5 * aa[value['column_required'][1]]) / aa[value['column_required'][2]]
                aa = aa.drop(columns = value['column_required'], axis = 1)
                key_name = value['Parameter_Name']
            elif value['method_of_compute'] == 'ts_pct_calc':
                aa = temp_df.groupby('player_id')[value['column_required']].rolling(value['rolling_window_required']).sum().shift().reset_index()
                aa[value['Parameter_Name']] = aa[value['column_required'][0]] /(2 * ( aa[value['column_required'][1]] + .44 * aa[value['column_required'][2]]))
                aa = aa.drop(columns = value['column_required'], axis = 1)
                key_name = value['Parameter_Name']
            poww = poww.append(aa)
        poww = poww.rename(columns={key_name: key})

        player_stats = player_stats.reset_index().merge(poww, how='left', on=['player_id', 'GAME-ID'])

    player_stats.to_csv(output_dir / dataset_config['NBA_Player_stats_w_rolling_avg'][i], index= False)

    i += 1
