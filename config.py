"""
A configuration file for some key settings.
"""


class Config:
    def __init__(self):
        # Start year for collecting data.
        self._start_year = 2000

        # Number of teams in the fantasy league.
        self._num_teams = 8

        # Number of players per team.
        self._team_size = 13

        # The years that will be used for model training.
        self._training_years = [
            2001,
            2003,
            2005,
            2006,
            2007,
            2009,
            2011,
            2012,
            2013,
            2015,
            2017,
        ]

        # The years that will be used for model testing.
        self._test_years = [2016, 2010, 2004]

        # The years that will be used for model validation.
        self._validation_years = [2014, 2008, 2002]

        # Number of training simulations to run per season.
        self._simulations_per_season = 10000

        # Column names for feature vector (and label).
        self._vector_columns = [
            "Age",
            "G",
            "GS",
            "MP/game",
            "FG/game",
            "FGA/game",
            "3P/game",
            "FT/game",
            "FTA/game",
            "TRB/game",
            "AST/game",
            "STL/game",
            "BLK/game",
            "TOV/game",
            "PTS/game",
            "total_fantasy_pts",
        ]

        # If True, do train-test-val split from all seasons. Otherwise, use the defined seasons for train-test-val.
        self._combine_data = False

    @property
    def start_year(self):
        return self._start_year

    @property
    def num_teams(self):
        return self._num_teams

    @property
    def team_size(self):
        return self._team_size

    @property
    def training_years(self):
        return self._training_years

    @property
    def test_years(self):
        return self._test_years

    @property
    def validation_years(self):
        return self._validation_years

    @property
    def simulations_per_season(self):
        return self._simulations_per_season

    @property
    def vector_columns(self):
        return self._vector_columns

    @property
    def combine_data(self):
        return self._combine_data
