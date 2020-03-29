import pandas as pd
import numpy as np
import yaml

dir = '/Users/Liu/NBA_Pro_Line_Analytics/'

with open(dir + 'config.yaml', 'r') as stream:
    try:
        config = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


def outcome_maker(df):
    if abs(df['SCORE_DIFF']) <= 5:
        return 0
    elif df['SCORE_DIFF'] >= 6:
        return 1
    elif df['SCORE_DIFF'] <= -6:
        return -1


def winning_team(df):
    if df['SCORE_DIFF'] < 0:
        return df['TEAM_RT']
    else:
        return df['TEAM_HT']


def losing_team(df):
    if df['SCORE_DIFF'] < 0:
        return df['TEAM_HT']
    else:
        return df['TEAM_RT']


def team_won(df):
    if df['Winning_Team'] == team:
        return 1
    else:
        return 0

def score_diff(df):
    if df[team + '_won'] == 0:
        return -1 * abs(df['SCORE_DIFF'])
    else:
        return abs(df['SCORE_DIFF'])


def outcome_equal_0(df):
    if df['Outcome'] == 0:
        return 1
    else:
        return 0


# Counts number of times the outcome equals 1
def outcome_equal_1(df):
    if ((df['TEAM_HT'] == team) & (df['Outcome'] == 1)):
        return 1
    elif ((df['TEAM_RT'] == team) & (df['Outcome'] == -1)):
        return 1
    else:
        return 0


# Counts number times the outcome equals -1
def outcome_equal_neg_1(df):
    if ((df['TEAM_HT'] == team) & (df['Outcome'] == -1)):
        return 1
    elif ((df['TEAM_RT'] == team) & (df['Outcome'] == 1)):
        return 1
    else:
        return 0

for data in config['NBA_Team_Cleaned_Data']:

    df_new = pd.read_csv(dir + 'build_data/NBA_Team_Stats_2010-2019/{0}'.format(data))
    home_teams = df_new[df_new.VENUE == 'H']
    road_teams = df_new[df_new.VENUE == 'R']
    match_df = home_teams.merge(road_teams, how='left', on=['GAME-ID',
                                                            'DATE',
                                                            'DATASET'], suffixes=('_HT', '_RT'))

    match_df['SCORE_DIFF'] = match_df['PTS_HT'] - match_df['PTS_RT']
    match_df['Outcome'] = match_df.apply(outcome_maker, axis=1)
    match_df['Winning_Team'] = match_df.apply(winning_team, axis=1)
    for team in match_df.TEAM_HT.unique():
        match_df[team + '_won'] = match_df.apply(team_won, axis=1)

    with open(dir + 'joined_params_config.yml', 'r') as stream:
        try:
            joined_param = yaml.load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    for namekey, value in joined_param.items():
        match_df['HT_' + namekey] = ''
        match_df['RT_' + namekey] = ''
        for team in match_df.TEAM_HT.unique():
            aa = match_df[(match_df.TEAM_HT == team) |
                          (match_df.TEAM_RT == team)][team + '_won'].rolling(
                value['rolling_window_required']).sum().shift().to_frame()
            aa_ht = aa.loc[np.where(match_df.TEAM_HT == team)]
            aa_rt = aa.loc[np.where(match_df.TEAM_RT == team)]
            key = team + '_won'
            aa_ht = aa_ht.rename(columns={key: 'HT_' + namekey})
            match_df.update(aa_ht)
            aa_rt = aa_rt.rename(columns={key: 'RT_' + namekey})
            match_df.update(aa_rt)

    data_name = data[0:len(data) - 4] + '_v2.csv'
    match_df.to_csv(dir + 'build_data/NBA_Team_Stats_2010-2019_Joined/{0}'.format(data_name))




dir = '/Users/Liu/NBA_Pro_Line_Analytics/'

with open(dir + 'joined_params_config_pt_diff.yml', 'r') as stream:
    try:
        config_joined_params_2  = yaml.load(stream)
    except yaml.YAMLError as exc:
        print(exc)


for df_data in config['NBA_Team_Cleaned_Data_v2']:
    data = pd.read_csv(dir + df_data)
    data['TOT_Final_Score'] = data['Final_Score_HT'] + data['Final_Score_RT']
    for key, value in config_joined_params_2.items():
        data['HT_' + key] = ""
        data['RT_' + key] = ""
        for team in data.TEAM_HT.unique():
            aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['TEAM_HT', 'TEAM_RT', 'Outcome', 'SCORE_DIFF',
                                                                        'Final_Score_HT', 'Final_Score_RT',
                                                                        team + '_won']]
            aa[team + '_scorediff'] = aa.apply(score_diff, axis=1)
            aa[team + '_' + key] = aa[team + '_scorediff'].rolling(value['rolling_window_required']).mean().shift()
            popo = team + '_' + key
            aa_ht = aa[aa.TEAM_HT == team]
            aa_ht = aa_ht.rename(columns={popo: 'HT_' + key})
            data.update(aa_ht['HT_' + key])
            aa_rt = aa[aa.TEAM_RT == team]
            aa_rt = aa_rt.rename(columns={popo: 'RT_' + key})
            data.update(aa_rt['RT_' + key])
    data.to_csv('/Users/Liu/NBA_Pro_Line_Analytics/build_data/NBA_Team_Stats_2010-2019_Joined/' + df_data[0:len(
        df_data) - 5] + '3.csv')

for df_data in config['NBA_Team_Cleaned_Data_v3']:
    data = pd.read_csv(dir + 'build_data/NBA_Team_Stats_2010-2019_Joined/' + df_data)
    for key, value in config_joined_params_3.items():
        data['HT_' + 'within_5_' + key] = ""
        data['RT_' + 'within_5_' + key] = ""
        data['HT_' + 'win_6+_' + key] = ""
        data['RT_' + 'win_6+_' + key] = ""
        data['HT_' + 'lose_6+_' + key] = ""
        data['RT_' + 'lose_6+_' + key] = ""
        for team in data.TEAM_HT.unique():
            aa = data[(data.TEAM_HT == team) | (data.TEAM_RT == team)][['DATE', 'TEAM_HT', 'TEAM_RT',
                                                                        'Final_Score_HT', 'Outcome']]
            aa['Outcome_0'] = aa.apply(outcome_equal_0, axis=1)
            aa['Outcome_1'] = aa.apply(outcome_equal_1, axis=1)
            aa['Outcome_neg_1'] = aa.apply(outcome_equal_neg_1, axis=1)
            aa[team + '_within_5_' + key] = aa['Outcome_0'].rolling(value['rolling_window_required']).sum().shift()
            aa[team + '_win_6+_' + key] = aa['Outcome_1'].rolling(value['rolling_window_required']).sum().shift()
            aa[team + '_lose_6+_' + key] = aa['Outcome_neg_1'].rolling(value['rolling_window_required']).sum().shift()
            outc_0 = team + '_within_5_' + key
            outc_1 = team + '_win_6+_' + key
            outc_neg1 = team + '_lose_6+_' + key
            aa_ht = aa[aa.TEAM_HT == team]
            aa_ht = aa_ht.rename(columns={outc_0: 'HT_' + 'within_5_' + key,
                                          outc_1: 'HT_' + 'win_6+_' + key,
                                          outc_neg1: 'HT_' + 'lose_6+_' + key})
            data.update(aa_ht[['HT_' + 'within_5_' + key, 'HT_' + 'win_6+_' + key, 'HT_' + 'lose_6+_' + key]])

            aa_rt = aa[aa.TEAM_RT == team]
            aa_rt = aa_rt.rename(columns={outc_0: 'RT_' + 'within_5_' + key,
                                          outc_1: 'RT_' + 'win_6+_' + key,
                                          outc_neg1: 'RT_' + 'lose_6+_' + key})
            data.update(aa_rt[['RT_' + 'within_5_' + key, 'RT_' + 'win_6+_' + key, 'RT_' + 'lose_6+_' + key]])
    data.to_csv('/Users/Liu/NBA_Pro_Line_Analytics/build_data/NBA_Team_Stats_2010-2019_Joined_Model_Build/' + df_data[0:len(
        df_data) - 5] + '4.csv',
                index=False)

