"""
The purpose of this class is to calculate rotesserie scores for a given set of players in a season.
For this model, we will assume standard 9-category Roto scoring:
https://en.wikipedia.org/wiki/Fantasy_basketball#Rotisserie_(ROTO)
Categories are: PTS, TRB, AST, TOV, FT%, FG%, 3P, STL, BLK
In a N-team league, a team will get N points for finishing 1st in a category, and 1 point for finishing last.
The team that accumulates the most points wins.
"""
from collections import defaultdict
import random


class RotoCalculator:
    def __init__(self, season_data):
        """
        Constructor.
        :param season_data: A dictionary with players as keys and their attributes as values.
        """
        self.season_data = season_data

    def calculate_team_stats(self, team):
        """
        Given a team, calculate its cumulative stats for the 9 categories.
        :param team: A list of player names.
        :return: A dictionary with the 9 stats as keys, and cumulative values as values.
        """
        team_stats = defaultdict(int)
        for player in team:
            try:
                player_info = self.season_data[player]
            except LookupError as e:
                raise LookupError(f"Player {player} not found in dataset.")
            stats = [
                "FG",
                "FGA",
                "FT",
                "FTA",
                "3P",
                "STL",
                "AST",
                "TRB",
                "TOV",
                "BLK",
                "PTS",
            ]
            for stat in stats:
                team_stats[stat] += player_info[stat]
        team_stats["FT%"] = team_stats["FT"] / team_stats["FTA"]
        team_stats["FG%"] = team_stats["FG"] / team_stats["FGA"]
        for cat in ["FG", "FGA", "FT", "FTA"]:
            team_stats.pop(cat)
        return team_stats

    def randomly_pick_teams(self, num_teams, team_size):
        """
        Randomly generate teams from the sample players.
        :param num_teams: The number of teams in the league.
        :param team_size: The size of each team.
        :return: A list of lists of player names.
        """
        sample_players = random.sample(list(self.season_data), num_teams * team_size)
        teams = [
            sample_players[i : i + team_size]
            for i in range(0, len(sample_players), team_size)
        ]
        return teams

    def run_simulation(self, num_teams=8, team_size=13):
        """
        Runs a ROTO full-season simulation for a fantasy league with randomly selected players.
        :param num_teams: The number of teams in the league.
        :param team_size: The size of each team.
        :return: A dictionary, with the five keys being:
            - 'team' - a list of player names on the team.
            - 'total_fantasy_pts' - the total amount of fantasy points scored.
                Max can receive num_teams number of points for winning each category.
            - 'rankings' - individual rankings for each stat. Highest rank is num_teams.
            - 'stats' - the cumulative stats that the team achieved during the season for each category.
            - 'overall_rank' - the overall rank of the team. 1 being highest, num_teams being the lowest.
        """
        teams = self.randomly_pick_teams(num_teams=num_teams, team_size=team_size)

        all_team_info = []
        for team in teams:
            team_info = {}
            team_info["team"] = team
            team_info["rankings"] = {}
            team_info["total_fantasy_pts"] = 0
            team_info["stats"] = self.calculate_team_stats(team)
            all_team_info.append(team_info)
        stats = ["FG%", "FT%", "3P", "STL", "AST", "TRB", "TOV", "BLK", "PTS"]
        for stat in stats:
            reverse = (
                False if stat == "TOV" else True
            )  # Ascending for turnover, Descending for everything else.
            sorted_list = sorted(
                all_team_info, key=lambda k: k["stats"][stat], reverse=reverse
            )
            for index, team in enumerate(sorted_list):
                points = num_teams - index
                team["rankings"][stat] = points
                team["total_fantasy_pts"] += points
        # Now calculate the overall ranking for each of the teams.
        sorted_list = sorted(
            all_team_info, key=lambda k: k["total_fantasy_pts"], reverse=True
        )
        for index, team in enumerate(sorted_list):
            team["overall_rank"] = index + 1
        return all_team_info
