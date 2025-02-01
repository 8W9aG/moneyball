"""The offensive efficiency feature extractor."""

# pylint: disable=too-many-return-statements
import pandas as pd
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.league_model import DELIMITER  # type: ignore
from sportsball.data.team_model import ASSISTS_COLUMN  # type: ignore
from sportsball.data.team_model import (FIELD_GOALS_ATTEMPTED_COLUMN,
                                        FIELD_GOALS_COLUMN,
                                        OFFENSIVE_REBOUNDS_COLUMN,
                                        TURNOVERS_COLUMN)

from .columns import (find_player_count, find_team_count, player_column_prefix,
                      team_column_prefix)
from .feature import Feature

OFFENSIVE_EFFICIENCY_COLUMN = "offensive_efficiency"


def _calculate_offensive_efficiency(row: pd.Series, col_prefix: str) -> float | None:
    field_goal_col = DELIMITER.join([col_prefix, FIELD_GOALS_COLUMN])
    if field_goal_col not in row:
        return None
    field_goals = row[field_goal_col]
    if field_goals is None:
        return None
    assists_col = DELIMITER.join([col_prefix, ASSISTS_COLUMN])
    if assists_col not in row:
        return None
    assists = row[assists_col]
    if assists is None:
        return None
    field_goals_attempted_col = DELIMITER.join(
        [col_prefix, FIELD_GOALS_ATTEMPTED_COLUMN]
    )
    if field_goals_attempted_col not in row:
        return None
    field_goals_attempted = row[field_goals_attempted_col]
    if field_goals_attempted is None:
        return None
    offensive_rebounds_col = DELIMITER.join([col_prefix, OFFENSIVE_REBOUNDS_COLUMN])
    if offensive_rebounds_col not in row:
        return None
    offensive_rebounds = row[offensive_rebounds_col]
    if offensive_rebounds is None:
        return None
    turnovers_col = DELIMITER.join([col_prefix, TURNOVERS_COLUMN])
    if turnovers_col not in row:
        return None
    turnovers = row[turnovers_col]
    if turnovers is None:
        return None
    return (float(field_goals) + float(assists)) / (
        float(field_goals_attempted)
        - float(offensive_rebounds)
        + float(assists)
        + float(turnovers)
    )


def _process_offensive_efficiency_df(df: pd.DataFrame) -> pd.DataFrame:
    team_count = find_team_count(df)
    player_count = find_player_count(df, team_count)

    columns = []
    for i in range(team_count):
        team_col = DELIMITER.join([team_column_prefix(i), OFFENSIVE_EFFICIENCY_COLUMN])
        df[team_col] = None
        columns.append(team_col)
        for j in range(player_count):
            player_col = DELIMITER.join(
                [player_column_prefix(i, j), OFFENSIVE_EFFICIENCY_COLUMN]
            )
            df[player_col] = None
            columns.append(player_col)

    def record_offensive_efficiency(row: pd.Series) -> pd.Series:
        for i in range(team_count):
            col_prefix = team_column_prefix(i)
            row[DELIMITER.join([col_prefix, OFFENSIVE_EFFICIENCY_COLUMN])] = (
                _calculate_offensive_efficiency(row, col_prefix)
            )
            for j in range(player_count):
                col_prefix = player_column_prefix(i, j)
                row[DELIMITER.join([col_prefix, OFFENSIVE_EFFICIENCY_COLUMN])] = (
                    _calculate_offensive_efficiency(row, col_prefix)
                )

        return row

    df = df.apply(record_offensive_efficiency, axis=1)
    df.attrs[str(FieldType.LOOKAHEAD)].extend(columns)
    return df


class OffensiveEfficiencyFeature(Feature):
    """The offensive efficiency feature extractor class."""

    # pylint: disable=too-few-public-methods

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process the dataframe and add the necessary features."""
        df = _process_offensive_efficiency_df(df)
        return df.copy()
