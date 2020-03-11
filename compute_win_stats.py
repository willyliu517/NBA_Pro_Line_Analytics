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
