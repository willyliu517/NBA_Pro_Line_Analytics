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

with open(config_dir + 'dataset_config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

with open(os.path.join(config_dir, 'team_metrics/pt_diff_params.yml'), 'r') as stream:
    try:
        config_pt_diff_params = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


scripts_dir = os.path.join(gbm_build_dir, 'model_build_scripts')
sys.path.insert(1, scripts_dir)

import helpers

def map_score_diff(df):
    if df['TEAM_HT'] == team:
        return df['score_diff']
    elif df['TEAM_RT'] == team:
        return -1 * df['score_diff']

output_dir = helpers.ensure_dir_exists(os.path.join(home_dir, 'NBA_Pro_Line_Analytics/', 'model_build_data/NBA_Team_Stats_2010-2019_Rolling_Avg_Win_Loss_pts_diff/'))


for df_data in config['NBA_Team_Stats_2010-2019_rolling_avg_win_loss']:
    data = pd.read_csv(os.path.join(home_dir,
                                    'NBA_Pro_Line_Analytics/model_build_data/NBA_Team_Stats_2010-2019_rolling_avg_win_loss/', df_data))
    data['TOT_Final_Score'] = data['Final_Score_HT'] + data['Final_Score_RT']
    for key, value in config_pt_diff_params.items():
        print(f'Now computing feature {key}')
        if key[0:10] == 'AVG_PTdiff':
            data['HT_' + key] = ""
            data['RT_' + key] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', 'outcome', 'score_diff','Final_Score_HT', 'Final_Score_RT']]
                aa[team + '_scorediff'] = aa.apply(map_score_diff, axis=1)
                aa[team + '_' + key] = aa[team + '_scorediff'].rolling(value['rolling_window_required']).mean().shift()
                popo = team + '_' + key
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
                data.update(aa_ht['HT_' + key])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
                data.update(aa_rt['RT_' + key])
        #Used to compute average point differential in wins
        elif key[0:14] == 'AVG_Win_PTdiff':
            data['HT_' + key] = ""
            data['RT_' + key] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', team + '_won']]
                aa[team + '_' + key] = aa[team + '_won'].rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_' + key
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
                data.update(aa_ht['HT_' + key])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
                data.update(aa_rt['RT_' + key])
        #Used to compute average point differential in losses
        elif key[0:15] == 'AVG_Loss_PTdiff':
            data['HT_' + key] = ""
            data['RT_' + key] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', team + '_lost']]
                aa[team + '_' + key] = aa[team + '_lost'].rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_' + key
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
                data.update(aa_ht['HT_' + key])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
                data.update(aa_rt['RT_' + key])
        #Used to compute number of times each outcome occured
        elif key[0:7] == 'outcome':
            rolling_window = key[11:]
            data['HT_cnt_within_5' + rolling_window] = ""
            data['RT_cnt_within_5' + rolling_window] = ""
            data['HT_cnt_wins_6_plus' + rolling_window] = ""
            data['RT_cnt_wins_6_plus' + rolling_window] = ""
            data['HT_cnt_loss_6_plus' + rolling_window] = ""
            data['RT_cnt_loss_6_plus' + rolling_window] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', 'score_diff', 'outcome_within_5']]
                aa[team + '_scorediff'] = aa.apply(map_score_diff, axis=1)
                #Caculates the number of outcomes within 5 for each team
                aa[team + '_' + 'outcome_within_5'] = aa['outcome_within_5'].rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_' + 'outcome_within_5'
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_cnt_within_5' + rolling_window})
                data.update(aa_ht['HT_cnt_within_5' + rolling_window])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_cnt_within_5' + rolling_window})
                data.update(aa_rt['RT_cnt_within_5' + rolling_window])
                #Caculates the number of wins with 6+ points for each team
                aa[team + '_win_6_plus'] = aa[team + '_scorediff'].apply(lambda x: 1 if x >= 6 else 0 ).rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_win_6_plus'
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_cnt_wins_6_plus' + rolling_window})
                data.update(aa_ht['HT_cnt_wins_6_plus' + rolling_window])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_cnt_wins_6_plus' + rolling_window})
                data.update(aa_rt['RT_cnt_wins_6_plus'+ rolling_window])
                #Caculates the number of losses with 6+ points for each team
                aa[team + '_loss_6_plus'] = aa[team + '_scorediff'].apply(lambda x: 1 if x <= -6 else 0 ).rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_loss_6_plus'
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_cnt_loss_6_plus' + rolling_window})
                data.update(aa_ht['HT_cnt_loss_6_plus' + rolling_window])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_cnt_loss_6_plus' + rolling_window})
                data.update(aa_rt['RT_cnt_loss_6_plus' + rolling_window])
        elif key[0:20] == 'num_11_plus_pts_wins':
            data['HT_' + key] = ""
            data['RT_' + key] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', 'score_diff']]
                aa[team + '_scorediff'] = aa.apply(map_score_diff, axis=1)
                aa[team + '_win_11_plus'] = aa[team + '_scorediff'].apply(lambda x: 1 if x >= 11 else 0 ).rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_win_11_plus'
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
                data.update(aa_ht['HT_' + key])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
                data.update(aa_rt['RT_' + key])
        elif key[0:20] == 'num_11_plus_pts_loss':
            data['HT_' + key] = ""
            data['RT_' + key] = ""
            for team in data.TEAM_HT.unique():
                aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', 'score_diff']]
                aa[team + '_scorediff'] = aa.apply(map_score_diff, axis=1)
                aa[team + '_lose_11_plus'] = aa[team + '_scorediff'].apply(lambda x: 1 if x <= -11 else 0 ).rolling(value['rolling_window_required']).sum().shift()
                popo = team + '_lose_11_plus'
                aa_ht = aa[aa.TEAM_HT == team]
                aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
                data.update(aa_ht['HT_' + key])
                aa_rt = aa[aa.TEAM_RT == team]
                aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
                data.update(aa_rt['RT_' + key])

    data_name = df_data[0:len(df_data) - 4] + '_pt_diff.csv'
    data.to_csv(os.path.join(output_dir, data_name), index = False)
