'''
Python script used to generate the features defined in the rolling_avg_metrics yaml file
'''

import pandas as pd
import yaml
import os

#Defines the directories
home_dir = '/Users/Liu/'
repo_dir = os.path.join(home_dir, 'NBA_Pro_Line_Analytics/')
config_dir = os.path.join(repo_dir,'GBM_model_build/feature_generation_scripts/configs/')

with open(config_dir + 'dataset_config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

# Opens the yaml containing parameter configurations
with open(config_dir + 'team_metrics/rolling_avg_metrics.yml', 'r') as stream:
    try:
        param_config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)

for data in config['NBA_Team_stats_raw']:
    print(f'Computing rolling averages on dataset {data}')
    df_new = pd.read_excel(repo_dir + 'raw_data/NBA_Team_Stats_2010-2019/{0}'.format(data), sheet_name=0)

    # The New Jersey Nets relocated to Brooklyn after the 2011-2012 season
    # Changing the name for more consistency
    df_new = df_new.replace('New Jersey', 'Brooklyn')

    # Drops the columns not used
    df_new = df_new.drop(['MAIN REF', 'CREW', 'HALFTIME', 'BOX SCORE\nURL', 'ODDS\nURL', 'OPENING ODDS',
                          'LINE \nMOVEMENT #1', 'LINE \nMOVEMENT #2', 'LINE \nMOVEMENT #3', 'TO',
                          'CLOSING\nODDS'], axis=1)
    # Naming conventions:
    # SF - Starting Forward
    # SG - Staring Guard
    # C - Center
    df_new = df_new.rename(columns={'STARTING LINEUPS': 'SF',
                                    'Unnamed: 38': 'SF2',
                                    'Unnamed: 39': 'C',
                                    'Unnamed: 40': 'SG2',
                                    'Unnamed: 41': 'SG1',
                                    'TO\nTO': 'Total_TO',
                                    'TEAM\nREST DAYS': 'Num_Rest_Days',
                                    'F': 'Final_Score'})

    # Calculates shooting percentages
    df_new = df_new.set_index(['GAME-ID'])
    #df_new['FT_PCT'] = df_new['FT'] / df_new['FTA']
    ##df_new['FG_PCT'] = df_new['FG'] / df_new['FGA']
    #df_new['3P_PCT'] = df_new['3P'] / df_new['3PA']
    # Formula for True Shooting Percentage from Wiki
    #df_new['TS_PCT'] = df_new['PTS'] / (2 * (df_new['FGA'] + .44 * df_new['FTA']))

    df_new['DATE'] = pd.to_datetime(df_new['DATE'])

    # Computes the rolling mean from the data
    for key, value in param_config.items():
        print(f'Now computing feature {key}')
        poww = pd.DataFrame()
        #Used for computing mean of features
        if value['method_of_compute'] == 'mean':
            for team in df_new.TEAM.unique():
                aa = df_new[df_new.TEAM == team].groupby('TEAM')[value['column_required']].rolling(
                    value['rolling_window_required']).mean().shift().reset_index()
                poww = poww.append(aa)
            df_new = df_new.reset_index()
            poww = poww.rename(columns={value['column_required']: value['Parameter_Name']})
            df_new = df_new.merge(poww, how='left', on=['TEAM', 'GAME-ID'])
            df_new = df_new.set_index(['GAME-ID'])
        #Used for computing percentage metrics
        elif value['method_of_compute'] == 'pct_calc':
            for team in df_new.TEAM.unique():
                aa = df_new[df_new.TEAM == team].groupby('TEAM')[value['column_required']].rolling(
                    value['rolling_window_required']).sum().shift().reset_index()
                aa[value['Parameter_Name']] = aa[value['column_required'][0]] / aa[value['column_required'][1]]
                poww = poww.append(aa)
            df_new = df_new.reset_index()
            poww = poww.drop(columns = value['column_required'], axis = 1)
            df_new = df_new.merge(poww, how='left', on=['TEAM', 'GAME-ID'])
            df_new = df_new.set_index(['GAME-ID'])
        #Used for computing true shooting metrics
        elif value['method_of_compute'] == 'ts_pct_calc':
            for team in df_new.TEAM.unique():
                aa = df_new[df_new.TEAM == team].groupby('TEAM')[value['column_required']].rolling(
                    value['rolling_window_required']).sum().shift().reset_index()
                aa[value['Parameter_Name']] = aa[value['column_required'][0]] / (2 * (0.44 * aa[value['column_required'][1]] + aa[value['column_required'][2]]))
                poww = poww.append(aa)
            df_new = df_new.reset_index()
            poww = poww.drop(columns = value['column_required'], axis = 1)
            df_new = df_new.merge(poww, how='left', on=['TEAM', 'GAME-ID'])
            df_new = df_new.set_index(['GAME-ID'])

    data_name = data[0:len(data) - 5] + '_w_rolling_avgs.csv'
    df_new.to_csv(repo_dir + 'model_build_data/NBA_Team_Stats_2010-2019/{0}'.format(data_name))
