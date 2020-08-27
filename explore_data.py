import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import TweedieRegressor, Ridge, SGDRegressor

def read_data(first_direc, second_direc, keep_cols, pos_id=1):
    raw_players_1 = pd.read_csv(first_direc)

    pos_players = pd.DataFrame(columns=raw_players_1.columns.tolist())

    for index, player in raw_players_1.iterrows():
        if player["element_type"] == pos_id and player['minutes'] > 0:
            pos_players = pos_players.append(player)

    raw_players_2 = pd.read_csv(second_direc)

    pos_players['cost'] = np.nan
    pos_players['y'] = np.nan


    for index, player in raw_players_2.iterrows():
        pos_players.loc[(pos_players['first_name'].str.lower() == player['first_name'].lower()) &
                    (pos_players['web_name'].str.lower() == player['web_name'].lower()),
                    ['cost', 'y', 'y_points']] = [player['now_cost'], player['total_points']/player['now_cost'], player['total_points']] 
        if player['web_name'] == "Sørloth":
            print(third_direc)
            print(player['web_name'])
            print(player['total_points'], "/", player['now_cost'])
            print("aguero: ",player['total_points']/player['now_cost'])


    remove_cols = [col for col in pos_players.columns if col not in keep_cols]
    pos_players = pos_players.drop(remove_cols, axis=1)
    pos_players = pos_players.dropna()
    names_inc = pos_players
    pos_players = pos_players.drop(['web_name'], axis=1)
    return pos_players, names_inc

if __name__ == "__main__":

    pd.set_option('display.max_rows', None)

    first_direc = "/Users/jonathanwindle/Documents/FPL/api_tests/data/2017-18/players_raw.csv"

    second_direc = "/Users/jonathanwindle/Documents/FPL/api_tests/data/2018-19/players_raw.csv"

    third_direc = "/Users/jonathanwindle/Documents/FPL/api_tests/data/2019-20/players_raw.csv"


    gk_keep_cols = ['web_name', 'minutes', 'penalties_saved', 'ict_index',
                 'goals_conceded', 'clean_sheets', 'saves', 'cost', 'total_points', 'y']

    fwd_keep_cols = ['web_name','minutes', 'goals_scored', 'assists', 'ict_index', 'total_points', 'cost', 'y']

    pos_id = 1

    keep_cols = gk_keep_cols

    
    gks, gks_names = read_data(first_direc, second_direc, keep_cols, pos_id)
    # print(gks)
    # print(len(gks))

    gks_test, gks_test_names = read_data(second_direc, third_direc, keep_cols, pos_id)

    # print(gks_test)

    gks_x = gks.iloc[:, :-1].values
    gks_y = gks.iloc[:,-1].values

    gks_x_test = gks_test.iloc[:, :-1].values
    gks_y_test = gks_test.iloc[:,-1].values

    scaler = StandardScaler()

    gks_x = scaler.fit_transform(gks_x)

    # reg = SVR(C=10, epsilon=0.2)

    reg = TweedieRegressor(power=1, alpha=0.5, link='log')

    reg.fit(gks_x, gks_y)

    gks_x_test = scaler.transform(gks_x_test)
    preds = reg.predict(gks_x_test)

    print(mean_squared_error(gks_y_test, preds))

    # print(gks_test_names)

    with open('gks.csv','w') as file:
        for idx, val in enumerate(preds):
            file.write(gks_test_names.iloc[idx]['web_name'] + "," + str(val) + "," + str(gks_y_test[idx]))
            file.write('\n')
