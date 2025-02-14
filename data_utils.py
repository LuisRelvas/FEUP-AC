import pandas as pd 
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# Read the data from the csv files
def load_data(): 
    csv_awardsPlayers = pd.read_csv('basketballPlayoffs/awards_players.csv')
    csv_coaches = pd.read_csv('basketballPlayoffs/coaches.csv')
    csv_playersTeams = pd.read_csv('basketballPlayoffs/players_teams.csv')
    csv_players = pd.read_csv('basketballPlayoffs/players.csv')
    csv_seriesPost = pd.read_csv('basketballPlayoffs/series_post.csv')
    csv_teamsPost = pd.read_csv('basketballPlayoffs/teams_post.csv')
    csv_teams = pd.read_csv('basketballPlayoffs/teams.csv')

    return csv_awardsPlayers, csv_coaches, csv_playersTeams, csv_players, csv_seriesPost, csv_teamsPost, csv_teams


def setup_seriesPost(csv): 
    csv = csv.drop(columns=['lgIDWinner', 'lgIDLoser'])
    return csv

def setup_coaches(csv_coaches):
    csv_coaches_aux = csv_coaches.copy()

    csv_coaches_aux = csv_coaches_aux.drop(csv_coaches_aux[(csv_coaches_aux['won'] == 0) & (csv_coaches_aux['lost'] == 0)].index)
    
    attributes_accumulative = ['won','lost', 'post_wins', 'post_losses']

    for attr in attributes_accumulative:
        csv_coaches_aux[attr] = csv_coaches_aux.groupby('coachID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).sum().fillna(0))
    
    csv_coaches_aux['winRateConf'] = np.where((csv_coaches_aux['won'] + csv_coaches_aux['lost']) > 0,
                                        csv_coaches_aux['won'] / (csv_coaches_aux['won'] + csv_coaches_aux['lost']),
                                        0)

    csv_coaches_aux['winRatePost'] = np.where((csv_coaches_aux['post_wins'] + csv_coaches_aux['post_losses']) > 0,
                                        csv_coaches_aux['post_wins'] / (csv_coaches_aux['post_wins'] + csv_coaches_aux['post_losses']),
                                        0)    
    
    csv_coaches_aux.drop(columns=['won','lost','post_wins','post_losses'], inplace=True)

    return csv_coaches_aux

def getCoachRating(csv_coaches, csv_teams):
    merged_data = csv_coaches.merge(csv_teams[['tmID', 'year', 'playoff']], on=['tmID', 'year'], how='left')
    relevant_columns = ['winRateConf', 'winRatePost','playoff'] 
    
    filtered_data = merged_data[relevant_columns]

    X = filtered_data.drop(columns=['playoff'])
    y = filtered_data['playoff']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1  
    best_model = None 
    best_coach_ratings = None  

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        filtered_data['coach_rating'] = model.predict(X)

        correlation = filtered_data['coach_rating'].corr(filtered_data['playoff'])
        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_coach_ratings = filtered_data[['coach_rating']].copy()

    scaler = MinMaxScaler(feature_range=(0, 99))
    best_coach_ratings[['coach_rating']] = scaler.fit_transform(best_coach_ratings[['coach_rating']])

    csv_coaches['coach_rating'] = best_coach_ratings['coach_rating']

    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100}%")
    return csv_coaches


def setup_teams(csv_teams,csv_teamsPost): 
        csv_teams = csv_teams.drop(columns=['lgID','divID','seeded','tmTRB','tmDRB','tmORB','opptmORB','opptmDRB','opptmTRB','homeW','homeL','awayW','awayL','confW','confL','attend','arena','name','franchID'])
        csv_teamsPost = csv_teamsPost.drop(columns=['lgID'])

        csv_merge = pd.merge(csv_teams,csv_teamsPost, on=['tmID','year'], how='left')

        csv_merge['W'] = csv_merge['W'].fillna(0)
        csv_merge['L'] = csv_merge['L'].fillna(0)

        attributes_mean = ["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","rank"]
        attributes_accumulative = ['W','L','won','lost']


        csv_merge = csv_merge.sort_values(by=['tmID', 'year'])

        for attr in attributes_mean:
            csv_merge[attr] = csv_merge.groupby('tmID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).mean().fillna(0))

        for attr in attributes_accumulative:
            csv_merge[attr] = csv_merge.groupby('tmID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).sum().fillna(0))

        csv_merge['winRateConf'] = np.where((csv_merge['won'] + csv_merge['lost']) > 0,
                                            csv_merge['won'] / (csv_merge['won'] + csv_merge['lost']),
                                            0)

        csv_merge['winRatePost'] = np.where((csv_merge['W'] + csv_merge['L']) > 0,
                                            csv_merge['W'] / (csv_merge['W'] + csv_merge['L']),
                                            0)

        csv_merge['playoff'] = csv_merge['playoff'].replace({'Y':1,'N':0})

        csv_merge.drop(columns=['W','L','won','lost'], inplace=True)


        attributes = [attr for attr in attributes_mean if attr.endswith('m') or attr.endswith('a')]

        attributes_m = [attr for attr in attributes_mean if attr.endswith('m')]
        attributes_a = [attr for attr in attributes_mean if attr.endswith('a')]

        for attr_m in attributes_m:
            attr_a = attr_m[:-1] + 'a'
            
            if attr_a in attributes_a:
                ratio_col = f'{attr_m[:-1]}_pct'
                
                csv_merge[ratio_col] = np.where(csv_merge[attr_a] > 0, csv_merge[attr_m] / csv_merge[attr_a], 0)
        
        csv_merge.drop(columns=attributes, inplace=True)


        csv_merge['o_oreb_pct'] = np.where(csv_merge['o_reb'] > 0, csv_merge['o_oreb'] / csv_merge['o_reb'], 0)
        csv_merge['o_dreb_pct'] = np.where(csv_merge['o_reb'] > 0, csv_merge['o_dreb'] / csv_merge['o_reb'], 0)

        csv_merge['d_oreb_pct'] = np.where(csv_merge['d_reb'] > 0, csv_merge['d_oreb'] / csv_merge['d_reb'], 0)
        csv_merge['d_dreb_pct'] = np.where(csv_merge['d_reb'] > 0, csv_merge['d_dreb'] / csv_merge['d_reb'], 0)

        csv_merge['asts_to_pct'] = np.where(csv_merge['o_to'] > 0, csv_merge['o_asts'] / csv_merge['o_to'], 0) 
        csv_merge['stl_to_pct'] = np.where(csv_merge['o_to'] > 0, csv_merge['o_stl'] / csv_merge['o_to'], 0)    

        csv_merge['d_asts_to_pct'] = np.where(csv_merge['d_to'] > 0, csv_merge['d_asts'] / csv_merge['d_to'], 0)
        csv_merge['d_stl_to_pct'] = np.where(csv_merge['d_to'] > 0, csv_merge['d_stl'] / csv_merge['d_to'], 0)

        csv_merge.drop(columns=['o_reb','o_dreb','o_oreb','d_oreb','d_reb','o_to','o_asts','o_stl','d_to','d_asts','d_to','d_stl'], inplace=True)

        csv_merge['post_rank'] = np.where(csv_merge['firstRound'] == 'L', 8,
                                        np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'L'), 4,
                                                np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'W') & (csv_merge['finals'] == 'L'), 2,
                                                        np.where((csv_merge['firstRound'] == 'W') & (csv_merge['semis'] == 'W') & (csv_merge['finals'] == 'W'), 1, 0))))



        csv_merge['post_rank'] = csv_merge['post_rank'].replace(0, 9)

        csv_merge['post_rank_cummean'] = csv_merge.groupby('tmID')['post_rank'].transform(
            lambda group: group.shift(1).rolling(window=2, min_periods=2).mean().fillna(0)
)

        csv_merge.drop(columns=['firstRound','semis','finals'], inplace=True)


        csv_teams_copy = csv_merge.copy()

        csv_teams_copy = csv_teams_copy.drop(columns=['year', 'confID', 'tmID'])

        correlation_with_playoff = csv_teams_copy.corr()['playoff'].drop('playoff')  

        print("Correlation with Playoff:")
        for feature, correlation in correlation_with_playoff.items():
            print(f"{feature}: {correlation * 100}%")

        correlation_matrix = correlation_with_playoff.values.reshape(1, -1)
        features = correlation_with_playoff.index

        plt.figure(figsize=(14, 2)) 
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', 
                    square=False, cbar_kws={"shrink": .8}, linewidths=0.5,
                    yticklabels=['playoff'], xticklabels=features)

        plt.title('Correlation with Playoff Heatmap')
        plt.xlabel('Features')
        plt.ylabel('Target: Playoff')
        plt.show()

        csv_aux = getTeamRating(csv_merge)


        correlation_with_playoff = csv_aux[['playoff', 'team_rating']].corr().loc['playoff'].drop('playoff')

        print("Correlation with Playoff:")
        for feature, correlation in correlation_with_playoff.items():
            print(f"{feature}: {correlation * 100:.2f}%")


        csv_teams = csv_merge.drop(columns=["o_pf","o_blk","o_pts","d_dreb","d_pf","d_blk","d_pts","GP","min","o_fg_pct","o_ft_pct","o_3p_pct","d_fg_pct","d_ft_pct","d_3p_pct","o_oreb_pct","o_dreb_pct","d_oreb_pct","d_dreb_pct","asts_to_pct","stl_to_pct","d_asts_to_pct","d_stl_to_pct"])

        csv_aux = csv_aux.drop(columns=["o_pf","o_blk","o_pts","d_dreb","d_pf","d_blk","d_pts","GP","min","o_fg_pct","o_ft_pct","o_3p_pct","d_fg_pct","d_ft_pct","d_3p_pct","o_oreb_pct","o_dreb_pct","d_oreb_pct","d_dreb_pct","asts_to_pct","stl_to_pct","d_asts_to_pct","d_stl_to_pct"])

        csv_teams = csv_aux

        return csv_teams

def setup_players(csv_players, csv_playersTeams): 
    csv_players = csv_players.drop(columns=['firstseason', 'lastseason'])
    players = csv_players['bioID'].unique()
    playersTeams = csv_playersTeams['playerID'].unique()
    csv_players.drop(csv_players[~csv_players['bioID'].isin(playersTeams)].index, inplace=True)

    return csv_players

def setup_players_algorithm(csv_players):
    print(f"Number of missing weights: {(csv_players['weight'] == 0).sum()}")

    important_columns = ['pos', 'height', 'weight']
    players_data = csv_players[important_columns]

    players_data = pd.get_dummies(players_data, columns=['pos'])

    known_weights = players_data[players_data['weight'] != 0]
    missing_weights = players_data[players_data['weight'] == 0]

    X_train = known_weights.drop(columns=['weight'])
    y_train = known_weights['weight']
    X_test = missing_weights.drop(columns=['weight'])

    models = {
        'KNN': KNeighborsRegressor(),
        'LinearRegression': LinearRegression(),
        'DecisionTree': DecisionTreeRegressor(),
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    best_model = None
    # Put here an infinte value
    best_score = float('-inf')
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
        avg_score = np.mean(scores)
        print(f"{name} model average MSE: {-avg_score}")
        
        if avg_score > best_score:
            best_score = avg_score
            best_model = model

    print(f"Best model selected: {best_model}")

    if isinstance(best_model, KNeighborsRegressor):
        initial_n_neighbors = int(np.sqrt(len(X_train)))
        param_grid = {'n_neighbors': range(1, initial_n_neighbors + 1)}
        grid_search = GridSearchCV(best_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    elif isinstance(best_model, DecisionTreeRegressor) or isinstance(best_model, RandomForestRegressor):
        param_grid = {'max_depth': range(1, 21)}
        grid_search = GridSearchCV(best_model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

    best_model.fit(X_train, y_train)
    predicted_weights = best_model.predict(X_test)

    csv_players.loc[csv_players['weight'] == 0, 'weight'] = predicted_weights

    print(f"Number of weights predicted: {len(predicted_weights)}")

    return csv_players


def setup_playersTeams(csv_playersTeams,csv_awardsPlayers):
    csv_playersTeams = csv_playersTeams.drop(columns=['lgID'])
    csv_playersTeams = csv_playersTeams.drop(csv_playersTeams[csv_playersTeams['GP'] == 0].index)
    csv_playersTeams = csv_playersTeams.drop(csv_playersTeams[csv_playersTeams['minutes'] == 0].index)

    attributes_mean = ["GP","GS","minutes","points","oRebounds","dRebounds","rebounds","assists","steals","blocks","turnovers","PF","fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted","threeMade","dq","PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds","PostdRebounds","PostRebounds","PostAssists","PostSteals","PostBlocks","PostTurnovers","PostPF","PostfgAttempted","PostfgMade","PostftAttempted","PostftMade","PostthreeAttempted","PostthreeMade","PostDQ"]
    
    for attr in attributes_mean:

        csv_playersTeams[attr] = csv_playersTeams.groupby('playerID')[attr].transform(lambda group: group.shift(1).rolling(min_periods=2, window=2).mean().fillna(0))
    
    csv_playersTeams['player_award_count'] = 0

    player_award_total = {}

    for index, row in csv_awardsPlayers.iterrows():
        player_id = row['playerID']
        
        if player_id not in player_award_total:
            player_award_total[player_id] = 0
        
        player_award_total[player_id] += 1
        
        csv_playersTeams.loc[
            (csv_playersTeams['playerID'] == player_id) & 
            (csv_playersTeams['year'] == row['year'] + 1),
            'player_award_count'
        ] = player_award_total[player_id] 


    csv_playersTeams = csv_playersTeams.sort_values(['year','tmID','playerID'])

    return csv_playersTeams


def getTeamRating(csv_teams):
    csv_teams_copy = csv_teams.copy()  

    X = csv_teams_copy.drop(columns=['playoff'])  
    y = csv_teams_copy['playoff']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1  
    best_model = None 
    best_team_ratings = None 

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        csv_teams_copy['team_rating'] = model.predict(X)

        correlation = csv_teams_copy['team_rating'].corr(csv_teams_copy['playoff'])

        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_team_ratings = csv_teams_copy[['team_rating']].copy()

    scaler = MinMaxScaler(feature_range=(0, 99))
    best_team_ratings[['team_rating']] = scaler.fit_transform(best_team_ratings[['team_rating']])

    csv_teams_copy['team_rating'] = best_team_ratings['team_rating']

    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100}%")

    return csv_teams_copy

def setup_awardsPlayers(csv_awardsPlayers): 
    csv_awardsPlayers = csv_awardsPlayers.drop(columns=['lgID'])
    csv_awardsPlayers['award'] = csv_awardsPlayers['award'].str.lower()
    return csv_awardsPlayers

def setup_teamsPost(csv_teamsPost): 
    csv_teamsPost = csv_teamsPost.drop(columns=['lgID'])
    return csv_teamsPost

def getPlayerRating(csv_playersTeams, csv_players, csv_teams):
    merged_data = csv_playersTeams.merge(csv_teams[['tmID', 'year', 'playoff']], on=['tmID', 'year'], how='left')

    relevant_columns = ['GP', 'GS', 'minutes', 'points', 'oRebounds', 'dRebounds', 'rebounds', 
                        'assists', 'steals', 'blocks', 'turnovers', 'PF', 
                        'fgAttempted', 'fgMade', 'ftAttempted', 'ftMade', 
                        'threeAttempted', 'threeMade', 'player_award_count', 'playoff']
    
    filtered_data = merged_data[relevant_columns]

    X = filtered_data.drop(columns=['playoff'])  
    y = filtered_data['playoff']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression(),
        "Support Vector Regression": SVR(),
        "Decision Tree": DecisionTreeRegressor(random_state=42)
    }

    best_correlation = -1 
    best_model = None 
    best_player_ratings = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        filtered_data['player_rating'] = model.predict(X)

        correlation = filtered_data['player_rating'].corr(filtered_data['playoff'])
        print(f"{model_name} Correlation with Playoff: {correlation * 100:.2f}%")

        if correlation > best_correlation:
            best_correlation = correlation
            best_model = model_name
            best_player_ratings = filtered_data['player_rating']

    print(f"Best Model: {best_model} with Correlation: {best_correlation * 100}%")

    merged_data['player_rating'] = best_player_ratings

    return merged_data



