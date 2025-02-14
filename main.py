import data_utils 
import pandas as pd
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder


def main(): 
    csv_awardsPlayers,csv_coaches, csv_playersTeams, csv_players, csv_seriesPost, csv_teamsPost, csv_teams =  data_utils.load_data()
    pd.options.mode.use_inf_as_na = True

    csv_new_teams = pd.read_csv('teste/teams.csv')
    csv_new_players = pd.read_csv('teste/players_teams.csv')
    csv_new_coaches = pd.read_csv('teste/coaches.csv')

    uniques_tmID = pd.concat([csv_new_teams['tmID'],csv_new_players['tmID'], csv_new_coaches['tmID'],csv_teamsPost['tmID'], csv_teams['tmID'],csv_playersTeams['tmID'],csv_coaches['tmID'],csv_seriesPost['tmIDWinner'],csv_seriesPost['tmIDLoser']]).unique()
    unique_confID = pd.concat([csv_teams['confID'],csv_new_teams['confID']]).unique()
    unique_playoffID = pd.concat([csv_teams['playoff']]).unique()

    label_tmID = LabelEncoder()
    label_tmID.fit(uniques_tmID)

    csv_new_teams['tmID'] = label_tmID.transform(csv_new_teams['tmID'])
    csv_new_players['tmID'] = label_tmID.transform(csv_new_players['tmID'])
    csv_new_coaches['tmID'] = label_tmID.transform(csv_new_coaches['tmID'])
    csv_teamsPost['tmID'] = label_tmID.transform(csv_teamsPost['tmID'])
    csv_teams['tmID'] = label_tmID.transform(csv_teams['tmID'])
    csv_playersTeams['tmID'] = label_tmID.transform(csv_playersTeams['tmID'])
    csv_coaches['tmID'] = label_tmID.transform(csv_coaches['tmID'])
    csv_seriesPost['tmIDWinner'] = label_tmID.transform(csv_seriesPost['tmIDWinner'])
    csv_seriesPost['tmIDLoser'] = label_tmID.transform(csv_seriesPost['tmIDLoser'])


    label_confID = LabelEncoder()
    label_confID.fit(unique_confID)

    csv_teams['confID'] = label_confID.transform(csv_teams['confID'])
    csv_new_teams['confID'] = label_confID.transform(csv_new_teams['confID'])


    label_playoffID = LabelEncoder()
    label_playoffID.fit(unique_playoffID)

    csv_teams['playoff'] = label_playoffID.transform(csv_teams['playoff'])
    csv_series_post = data_utils.setup_seriesPost(csv_seriesPost)
    csv_teams = data_utils.setup_teams(csv_teams, csv_teamsPost)
    csv_players = data_utils.setup_players(csv_players, csv_playersTeams)
    csv_playersTeams = data_utils.setup_playersTeams(csv_playersTeams, csv_awardsPlayers)
    csv_coaches = data_utils.setup_coaches(csv_coaches)
    csv_awardsPlayers = data_utils.setup_awardsPlayers(csv_awardsPlayers)


    # Get the ratings for the teams, coaches and players
    csv_playerRating = data_utils.getPlayerRating(csv_playersTeams, csv_players, csv_teams)
    csv_coachRating = data_utils.getCoachRating(csv_coaches, csv_teams)
    csv_teamRating = data_utils.getTeamRating(csv_teams)
    # Modify the models just as you want
    teamModel = MLPRegressor()
    coachModel = MLPRegressor()
    squadModel = MLPRegressor()
    # Modify the year to get the predictions for that year
    year = 10
    baseline(csv_coachRating,csv_playerRating, csv_teamRating,csv_teams, teamModel, coachModel, squadModel, year, label_tmID)
    kaggle(csv_coachRating,csv_playerRating, csv_teamRating,csv_teams, csv_new_coaches, csv_new_teams, csv_new_players,teamModel, coachModel, squadModel, label_tmID)
    

def evaluation_metrics(csv):
    score = 0.0 
    for index, row in csv.iterrows(): 
        if row['playoff'] != row['predicted_playoff']: 
            score += 1
    print(f"The final score is {score}")
    return score
    
# In the current implementation it is not possible to get the same results as the ones in the submission. To get the exact predictions of the ones submitted in the Kaggle, please remove the grid search in the Regressor algorithms. 
# The algorithms used in the submission are the following:
# Linear Regression, Decision Tree Regressor, MLP Regressor
# In the first three predictions instead of using an model to predict the squad rating, we used the mean of the top 10 players that belong to that team.
# Below you have the last submission that we made, which was considered, alongside with the first one, the best.  
def kaggle(csv_coachesRating, csv_playersRating, csv_teamRating ,csv_teams, csv_new_coaches, csv_new_teams, csv_new_players,teamModel,coachModel,squadModel, label_tmID):

    csv_merged = pd.merge(csv_coachesRating, csv_teams, on=['tmID', 'year'], how='left')
    csv_merged_aux = csv_merged[['tmID', 'year', 'coach_rating', 'team_rating', 'playoff', 'confID']]

    top10_players = csv_playersRating.groupby(['tmID', 'year']).apply(lambda x: x.nlargest(10, 'player_rating')).reset_index(drop=True)

    top10_mean = top10_players.groupby(['tmID', 'year'])['player_rating'].last().reset_index(name='top10_mean')

    csv_merged_aux = csv_merged_aux.merge(top10_mean, on=['tmID', 'year'], how='left')

    csv_merged_aux = csv_merged_aux.groupby(['year','tmID']).last().reset_index()

    csv_merged_aux.rename(columns={'top10_mean': 'squad_rating'}, inplace=True)

    team_model = teamModel
    coach_model = coachModel
    squad_model = squadModel

    csv_merge_test = pd.merge(csv_new_coaches, csv_new_teams, on=['tmID', 'year'], how='left')
    csv_merge_test = pd.merge(csv_merge_test, csv_new_players, on=['tmID', 'year'], how='left')

    csv_merge_test = csv_merge_test.groupby(['year','tmID']).last().reset_index()

    csv_merge_test['playoff'] = 0
    csv_merge_test = csv_merge_test[['year','tmID','confID','playoff']]

    train = csv_merged_aux[(csv_merged_aux['year'] > 0) & (csv_merged_aux['year'] <= 10)].sort_values(['tmID', 'year'])
    test = csv_merge_test.sort_values(['tmID', 'year'])

    x_train_previous = train[['year', 'tmID']]
    y_train_previous = train[['team_rating', 'coach_rating', 'squad_rating']]
    x_test_previous = test[['year', 'tmID']]

    test['team_rating'] = 0
    test['coach_rating'] = 0
    test['squad_rating'] = 0
    y_test_previous = test[['team_rating', 'coach_rating']]

    classifiers = {
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(),
        'RandomForest': RandomForestClassifier(),
        'SVC': SVC(probability=True),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'GradientBoosting': GradientBoostingClassifier(),
    }

    results = []

    best_manual_accuracy = 0

    param_grid = {
        'KNN': {'n_neighbors': [3, 5], 'weights': ['uniform']},
        'DecisionTree': {'max_depth': [None, 10], 'min_samples_split': [2, 5]},
        'RandomForest': {'n_estimators': [100], 'max_depth': [None, 10]},
        'SVC': {'C': [1], 'kernel': ['linear'], 'gamma': ['scale']},
        'LogisticRegression': {'C': [1]},
        'GradientBoosting': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]},
        'NeuralNetwork': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'solver': ['adam'], 'max_iter': [200, 500]},
    }

    team_model.fit(x_train_previous, y_train_previous['team_rating'])
    test['team_rating'] = team_model.predict(x_test_previous)

    coach_model.fit(x_train_previous, y_train_previous['coach_rating'])
    test['coach_rating'] = coach_model.predict(x_test_previous)

    squad_model.fit(x_train_previous, y_train_previous['squad_rating'])
    test['squad_rating'] = squad_model.predict(x_test_previous)

    test = test.sort_values(['tmID', 'year'])

    x_test = test[['year','tmID','team_rating', 'coach_rating', 'confID', 'squad_rating']]
    y_test = test['playoff']
    x_train = train[['year','tmID','team_rating','coach_rating', 'confID', 'squad_rating']]
    y_train = train['playoff']

    def grid_search(clf_name, clf):
        grid_search = GridSearchCV(clf, param_grid[clf_name], cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        return clf_name, grid_search.best_estimator_

    best_classifiers = {}
    for clf_name, clf in classifiers.items():
        best_classifiers[clf_name] = grid_search(clf_name, clf)[1]

    voting_clf = VotingClassifier(estimators=[
        ('KNN', best_classifiers['KNN']),
        ('DecisionTree', best_classifiers['DecisionTree']),
        ('RandomForest', best_classifiers['RandomForest']),
        ('SVC', best_classifiers['SVC']),
        ('LogisticRegression', best_classifiers['LogisticRegression']),
        ('GradientBoosting', best_classifiers['GradientBoosting'])
    ], voting='soft') 

    voting_clf.fit(x_train, y_train)

    y_prob = voting_clf.predict_proba(x_test)[:, 1]

    y_prob = (y_prob / y_prob.sum()) * 8
    test['playoff_prob'] = y_prob

    test['predicted_playoff'] = test.groupby('confID')['playoff_prob'].rank(ascending=False, method='first') <= 4
    test['predicted_playoff'] = test['predicted_playoff'].astype(int)

    accuracy_score_value = accuracy_score(y_test, test['predicted_playoff'])
    manual_accuracy = (test['playoff'] == test['predicted_playoff']).mean() * 100

    results.append({
        'TeamModel': teamModel,
        'CoachModel': coachModel,
        'SquadModel': squadModel,
        'PlayoffModel': 'Ensemble',
        'AccuracyScore': accuracy_score_value * 100,
        'ManualAccuracy': manual_accuracy
    })

    if manual_accuracy > best_manual_accuracy:
        best_manual_accuracy = manual_accuracy
        best_team_model = teamModel
        best_coach_model = coachModel
        best_squad_model = squadModel
        best_clf = 'Ensemble'
        output = test.copy()

    results_df = pd.DataFrame(results)
    results_df.sort_values(by=['ManualAccuracy', 'AccuracyScore'], ascending=False, inplace=True)

    print("Model combinations and their accuracies have been saved to 'model_combinations_results.csv'")
    print(results_df.sort_values(by=['ManualAccuracy', 'AccuracyScore'], ascending=False))

    print(f"Best model: {best_team_model}, {best_coach_model}, {best_squad_model}, {best_clf} with manual accuracy of {best_manual_accuracy:.2f}%")
    output = output[['tmID','confID','playoff','playoff_prob','predicted_playoff']]

    output['tmIDString'] = label_tmID.inverse_transform(output['tmID'])
    output = output[['tmIDString','predicted_playoff']]
    output['tmID'] = output['tmIDString']
    output = output[['tmID','predicted_playoff']]
    output.to_csv('output_kaggle.csv', index=False)
    print(output)


def baseline(csv_coachRating, csv_playerRating, csv_teamRating, csv_teams, teamModel, coachModel, playerModel, year, label_tmID):
    random_state = 42
    csv_merged = pd.merge(csv_coachRating, csv_teams, on=['tmID', 'year'], how='left')
    csv_merged_aux = csv_merged[['tmID', 'year', 'coach_rating', 'team_rating', 'playoff', 'confID']]

    top10_players = csv_playerRating.groupby(['tmID', 'year']).apply(lambda x: x.nlargest(10, 'player_rating')).reset_index(drop=True)

    top10_mean = top10_players.groupby(['tmID', 'year'])['player_rating'].last().reset_index(name='top10_mean')

    csv_merged_aux = csv_merged_aux.merge(top10_mean, on=['tmID', 'year'], how='left')

    csv_merged_aux = csv_merged_aux.groupby(['year', 'tmID']).last().reset_index()

    csv_merged_aux.rename(columns={'top10_mean': 'squad_rating'}, inplace=True)

    team_model = teamModel
    coach_model = coachModel
    player_model = playerModel

    train = csv_merged_aux[(csv_merged_aux['year'] > 0) & (csv_merged_aux['year'] < year)].sort_values(['tmID', 'year'])
    test = csv_merged_aux[csv_merged_aux['year'] == year]

    x_train_previous = train[['year', 'tmID']]
    y_train_previous = train[['team_rating', 'coach_rating', 'squad_rating']]
    x_test_previous = test[['year', 'tmID']]

    test['team_rating'] = 0
    test['coach_rating'] = 0
    test['squad_rating'] = csv_merged_aux['squad_rating']
    y_test_previous = test[['team_rating', 'coach_rating']]

    regressor_param_grid = {
        'MLPRegressor': {'hidden_layer_sizes': [(50,), (100,)], 'activation': ['relu', 'tanh'], 'solver': ['adam'], 'max_iter': [200, 500]},
    }

    def grid_search_regressor(model, param_grid, X, y):
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
        grid_search.fit(X, y)
        return grid_search.best_estimator_

    team_model = grid_search_regressor(MLPRegressor(), regressor_param_grid['MLPRegressor'], x_train_previous, y_train_previous['team_rating'])
    coach_model = grid_search_regressor(MLPRegressor(), regressor_param_grid['MLPRegressor'], x_train_previous, y_train_previous['coach_rating'])

    test['team_rating'] = team_model.predict(x_test_previous)
    test['coach_rating'] = coach_model.predict(x_test_previous)

    team_train_predictions = team_model.predict(x_train_previous)
    coach_train_predictions = coach_model.predict(x_train_previous)

    team_train_rmse = np.sqrt(mean_squared_error(y_train_previous['team_rating'], team_train_predictions))
    coach_train_rmse = np.sqrt(mean_squared_error(y_train_previous['coach_rating'], coach_train_predictions))

    print(f"Team Model Training RMSE: {team_train_rmse}")
    print(f"Coach Model Training RMSE: {coach_train_rmse}")

    def plot_learning_curve(model, X, y, title):
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring='neg_mean_squared_error')
        train_scores_mean = -np.mean(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='Training error')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='orange', label='Cross-validation error')

        plt.title(title)
        plt.xlabel('Training')
        plt.ylabel('Mean Squared Error')
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    plot_learning_curve(team_model, x_train_previous, y_train_previous['team_rating'], "Team Model Learning Curve Season 10")
    plot_learning_curve(coach_model, x_train_previous, y_train_previous['coach_rating'], "Coach Model Learning Curve Season 10")

    classifiers = {
        'KNN': KNeighborsClassifier(),
        'DecisionTree': DecisionTreeClassifier(random_state=random_state),
        'SVC': SVC(probability=True, random_state=random_state),
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state),
    }

    param_grid = {
        'KNN': {'n_neighbors': [30], 'weights': ['uniform']},
        'DecisionTree': {'max_depth': [None, 100], 'min_samples_split': [2, 5, 7, 10]},
        'SVC': {'C': [1], 'kernel': ['linear'], 'gamma': ['scale']},
        'LogisticRegression': {'C': [1]},
        'GradientBoosting': {'n_estimators': [100], 'learning_rate': [0.1], 'max_depth': [3]},
    }

    x_test = test[['year', 'tmID', 'team_rating', 'coach_rating', 'confID', 'squad_rating']]
    y_test = test['playoff']
    x_train = train[['year', 'tmID', 'team_rating', 'coach_rating', 'confID', 'squad_rating']]
    y_train = train['playoff']

    def grid_search(clf_name, clf):
        grid_search = GridSearchCV(clf, param_grid[clf_name], cv=5, scoring='accuracy')
        grid_search.fit(x_train, y_train)
        return clf_name, grid_search.best_estimator_

    best_classifiers = {}
    for clf_name, clf in classifiers.items():
        best_classifiers[clf_name] = grid_search(clf_name, clf)[1]

    voting_clf = VotingClassifier(estimators=[
        ('KNN', best_classifiers['KNN']),
        ('DecisionTree', best_classifiers['DecisionTree']),
        ('SVC', best_classifiers['SVC']),
        ('LogisticRegression', best_classifiers['LogisticRegression']),
        ('GradientBoosting', best_classifiers['GradientBoosting'])
    ], voting='soft')

    voting_clf.fit(x_train, y_train)

    accuracies = {}
    for clf_name, clf in best_classifiers.items():
        model_predictions_prob = clf.predict_proba(x_test)[:, 1]
        model_predictions = (model_predictions_prob >= 0.5).astype(int)
        accuracies[clf_name] = accuracy_score(y_test, model_predictions) * 100

    y_prob = voting_clf.predict_proba(x_test)[:, 1]
    y_prob = (y_prob / y_prob.sum()) * 8
    test['playoff_prob'] = y_prob
    test['predicted_playoff'] = test.groupby('confID')['playoff_prob'].rank(ascending=False, method='first') <= 4
    test['predicted_playoff'] = test['predicted_playoff'].astype(int)

    accuracies['Ensemble'] = accuracy_score(y_test, test['predicted_playoff']) * 100

    plt.figure(figsize=(10, 6))
    plt.bar(accuracies.keys(), accuracies.values(), color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy of Individual Models and Ensemble')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    output = test[['tmID', 'confID', 'playoff', 'playoff_prob', 'predicted_playoff']]
    output['tmIDString'] = label_tmID.inverse_transform(output['tmID'])
    output = output[['tmIDString', 'confID', 'playoff', 'playoff_prob', 'predicted_playoff']]
    output.sort_values('playoff_prob', ascending=False, inplace=True)
    output.to_csv('output.csv', index=False)

    evaluation_metrics(output[['tmIDString', 'playoff', 'predicted_playoff']])



main()



