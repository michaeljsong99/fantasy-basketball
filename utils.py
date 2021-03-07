def calculate_team_prev_season_data(team, prev_season_data):
    """
    Return a list of size 15, where the items represent the following average attributes for the team:
    [Age G GS MP/game FG/game FGA/game 3P/game FT/game FTA/game TRB/game AST/game STL/game BLK/game TOV/game PTS/game]
    :param team: A list of player names.
    :param prev_season_data: A dictionary with player names as keys and their normalized prev season data as values.
    :return: A list of size 15.
    """
    num_players = len(team)
    team_info = {
        "Age": 0,
        "G": 0,
        "GS": 0,
        "MP/game": 0,
        "FG/game": 0,
        "FGA/game": 0,
        "3P/game": 0,
        "FT/game": 0,
        "FTA/game": 0,
        "TRB/game": 0,
        "AST/game": 0,
        "STL/game": 0,
        "BLK/game": 0,
        "TOV/game": 0,
        "PTS/game": 0,
    }
    stats = list(team_info.keys())
    for player in team:
        data = prev_season_data[player]
        for stat in stats:
            team_info[stat] += data[stat]

    # Create a list of the features.
    team_features = []
    for feature in stats:
        # Divide by the number of players on the team to get the average.
        team_features.append(team_info[feature] / num_players)
    return team_features
