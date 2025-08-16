"""The strategy class."""

# pylint: disable=too-many-statements,line-too-long,invalid-unary-operand-type,too-many-lines
import datetime
import functools
import hashlib
import json
import os
import pickle

import numpy as np
import optuna
import pandas as pd
import pytz
import wavetrainer as wt  # type: ignore
from sportsball.data.address_model import (ADDRESS_LATITUDE_COLUMN,
                                           ADDRESS_LONGITUDE_COLUMN)
from sportsball.data.bookie_model import BOOKIE_IDENTIFIER_COLUMN
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN  # type: ignore
from sportsball.data.game_model import VENUE_COLUMN_PREFIX
from sportsball.data.league_model import DELIMITER  # type: ignore
from sportsball.data.news_model import (NEWS_PUBLISHED_COLUMN,
                                        NEWS_SOURCE_COLUMN, NEWS_TITLE_COLUMN)
from sportsball.data.odds_model import (DT_COLUMN, ODDS_BET_COLUMN,
                                        ODDS_BOOKIE_COLUMN,
                                        ODDS_CANONICAL_COLUMN)
from sportsball.data.player_model import \
    FIELD_GOALS_ATTEMPTED_COLUMN as \
    PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN  # type: ignore
from sportsball.data.player_model import \
    FIELD_GOALS_COLUMN as PLAYER_FIELD_GOALS_COLUMN  # type: ignore
from sportsball.data.player_model import \
    OFFENSIVE_REBOUNDS_COLUMN as PLAYER_OFFENSIVE_REBOUNDS_COLUMN
from sportsball.data.player_model import (
    PLAYER_ACLI_COLUMN, PLAYER_AERIALS_LOST_COLUMN, PLAYER_AERIALS_WON_COLUMN,
    PLAYER_ASSIST_PERCENTAGE_COLUMN, PLAYER_ASSIST_TACKLES_COLUMN,
    PLAYER_ASSISTS_COLUMN, PLAYER_AT_BATS_COLUMN,
    PLAYER_AVERAGE_DISTANCE_OF_DEFENSIVE_ACTIONS_COLUMN,
    PLAYER_AVERAGE_GAIN_COLUMN, PLAYER_AVERAGE_GOAL_KICK_LENGTH_COLUMN,
    PLAYER_AVERAGE_INTERCEPTION_YARDS_COLUMN,
    PLAYER_AVERAGE_KICKOFF_RETURN_YARDS_COLUMN,
    PLAYER_AVERAGE_KICKOFF_YARDS_COLUMN, PLAYER_AVERAGE_LEVERAGE_INDEX_COLUMN,
    PLAYER_AVERAGE_PASS_LENGTH_COLUMN, PLAYER_AVERAGE_PUNT_RETURN_YARDS_COLUMN,
    PLAYER_AVERAGE_SACK_YARDS_COLUMN, PLAYER_AVERAGE_STUFF_YARDS_COLUMN,
    PLAYER_BALL_BATSMAN_RUNS_COLUMN, PLAYER_BALL_OVER_ACTUAL_COLUMN,
    PLAYER_BALL_OVER_UNIQUE_COLUMN, PLAYER_BALL_RECOVERIES_COLUMN,
    PLAYER_BALL_TOTAL_RUNS_COLUMN, PLAYER_BALLS_COLUMN,
    PLAYER_BASES_ON_BALLS_COLUMN, PLAYER_BATTERS_FACED_COLUMN,
    PLAYER_BATTING_STYLE_COLUMN, PLAYER_BEHINDS_COLUMN,
    PLAYER_BIRTH_DATE_COLUMN, PLAYER_BLOCK_PERCENTAGE_COLUMN,
    PLAYER_BLOCKED_FIELD_GOAL_TOUCHDOWNS_COLUMN,
    PLAYER_BLOCKED_PUNT_EZ_REC_TD_COLUMN,
    PLAYER_BLOCKED_PUNT_TOUCHDOWNS_COLUMN, PLAYER_BLOCKS_COLUMN,
    PLAYER_BOUNCES_COLUMN, PLAYER_BOWLING_STYLE_COLUMN,
    PLAYER_BOX_PLUS_MINUS_COLUMN, PLAYER_BROWNLOW_VOTES_COLUMN,
    PLAYER_CARRIES_COLUMN, PLAYER_CARRIES_INTO_FINAL_THIRD_COLUMN,
    PLAYER_CARRIES_INTO_PENALTY_AREA_COLUMN, PLAYER_CHALLENGES_LOST_COLUMN,
    PLAYER_CLANGERS_COLUMN, PLAYER_CLEARANCES_COLUMN,
    PLAYER_COMPLETION_PERCENTAGE_COLUMN, PLAYER_COMPLETIONS_COLUMN,
    PLAYER_CONCEDED_COLUMN, PLAYER_CONTESTED_MARKS_COLUMN,
    PLAYER_CONTESTED_POSSESSIONS_COLUMN, PLAYER_CORNER_KICKS_COLUMN,
    PLAYER_CORSI_FOR_PERCENTAGE_COLUMN, PLAYER_CROSSES_COLUMN,
    PLAYER_CROSSES_FACED_COLUMN, PLAYER_CROSSES_INTO_PENALTY_AREA_COLUMN,
    PLAYER_CROSSES_STOPPED_COLUMN, PLAYER_CWPA_COLUMN,
    PLAYER_DEAD_BALL_PASSES_COLUMN, PLAYER_DECISION_COLUMN,
    PLAYER_DEFENSIVE_ACTIONS_OUTSIDE_PENALTY_AREA_COLUMN,
    PLAYER_DEFENSIVE_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_DEFENSIVE_FUMBLE_RETURNS_COLUMN,
    PLAYER_DEFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN, PLAYER_DEFENSIVE_POINTS_COLUMN,
    PLAYER_DEFENSIVE_RATING_COLUMN, PLAYER_DEFENSIVE_REBOUND_PERCENTAGE_COLUMN,
    PLAYER_DEFENSIVE_REBOUNDS_COLUMN, PLAYER_DEFENSIVE_TOUCHDOWNS_COLUMN,
    PLAYER_DEFENSIVE_ZONE_STARTS_COLUMN, PLAYER_DISPOSALS_COLUMN,
    PLAYER_DISPOSSESSED_COLUMN, PLAYER_DOTS_COLUMN,
    PLAYER_DRIBBLERS_TACKLED_COLUMN, PLAYER_DRIBBLES_CHALLENGED_COLUMN,
    PLAYER_EARNED_RUNS_COLUMN, PLAYER_ECONOMY_COLUMN,
    PLAYER_EFFECTIVE_FIELD_GOAL_PERCENTAGE_COLUMN, PLAYER_ERA_COLUMN,
    PLAYER_ERRORS_COLUMN, PLAYER_ESPN_QUARTERBACK_RATING_COLUMN,
    PLAYER_ESPN_RUNNINGBACK_RATING_COLUMN, PLAYER_ESPN_WIDERECEIVER_COLUMN,
    PLAYER_EVEN_STRENGTH_ASSISTS_COLUMN, PLAYER_EVEN_STRENGTH_GOALS_COLUMN,
    PLAYER_EXPECTED_ASSISTED_GOALS_COLUMN, PLAYER_EXPECTED_ASSISTS_COLUMN,
    PLAYER_EXPECTED_GOALS_COLUMN, PLAYER_EXTRA_POINT_ATTEMPTS_COLUMN,
    PLAYER_EXTRA_POINT_BLOCKED_COLUMN, PLAYER_EXTRA_POINT_PERCENTAGE_COLUMN,
    PLAYER_EXTRA_POINTS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_EXTRA_POINTS_MADE_COLUMN, PLAYER_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_FAIR_CATCHES_COLUMN, PLAYER_FALL_OF_WICKET_BALLS_COLUMN,
    PLAYER_FALL_OF_WICKET_NUM_COLUMN, PLAYER_FALL_OF_WICKET_ORDER_COLUMN,
    PLAYER_FALL_OF_WICKET_OVER_NUMBER_COLUMN,
    PLAYER_FALL_OF_WICKET_OVERS_COLUMN, PLAYER_FALL_OF_WICKET_RUNS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPT_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_ABOVE_50_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_19_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_29_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_39_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_49_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_59_YARDS_COLUMN,
    PLAYER_FIELD_GOAL_ATTEMPTS_MAX_99_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_BLOCKED_COLUMN,
    PLAYER_FIELD_GOALS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_FIELD_GOALS_MADE_ABOVE_50_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_19_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_29_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_39_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_49_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_59_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_MAX_99_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MADE_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_MISSED_YARDS_COLUMN,
    PLAYER_FIELD_GOALS_PERCENTAGE_COLUMN, PLAYER_FLY_BALLS_COLUMN,
    PLAYER_FORCED_FUMBLES_COLUMN, PLAYER_FOULS_COMMITTED_COLUMN,
    PLAYER_FOULS_DRAWN_COLUMN, PLAYER_FOURS_COLUMN,
    PLAYER_FREE_KICKS_AGAINST_COLUMN, PLAYER_FREE_KICKS_FOR_COLUMN,
    PLAYER_FREE_THROW_ATTEMPT_RATE_COLUMN, PLAYER_FREE_THROWS_ATTEMPTED_COLUMN,
    PLAYER_FREE_THROWS_COLUMN, PLAYER_FREE_THROWS_PERCENTAGE_COLUMN,
    PLAYER_FUMBLE_RECOVERIES_COLUMN, PLAYER_FUMBLE_RECOVERY_YARDS_COLUMN,
    PLAYER_FUMBLES_COLUMN, PLAYER_FUMBLES_LOST_COLUMN,
    PLAYER_FUMBLES_RECOVERED_COLUMN, PLAYER_FUMBLES_RECOVERED_YARDS_COLUMN,
    PLAYER_FUMBLES_TOUCHDOWNS_COLUMN, PLAYER_GAME_SCORE_COLUMN,
    PLAYER_GAME_WINNING_GOALS_COLUMN, PLAYER_GOAL_ASSISTS_COLUMN,
    PLAYER_GOAL_CREATING_ACTIONS_COLUMN, PLAYER_GOAL_KICKS_ATTEMPTED_COLUMN,
    PLAYER_GOALS_AGAINST_COLUMN, PLAYER_GOALS_COLUMN,
    PLAYER_GROSS_AVERAGE_PUNT_YARDS_COLUMN, PLAYER_GROUND_BALLS_COLUMN,
    PLAYER_HANDBALLS_COLUMN, PLAYER_HEADSHOT_COLUMN, PLAYER_HEIGHT_COLUMN,
    PLAYER_HIT_OUTS_COLUMN, PLAYER_HITS_AT_BATS_COLUMN, PLAYER_HITS_COLUMN,
    PLAYER_HOME_RUNS_COLUMN, PLAYER_HURRIES_COLUMN,
    PLAYER_INDIVIDUAL_CORSI_FOR_EVENTS_COLUMN, PLAYER_INHERITED_RUNNERS_COLUMN,
    PLAYER_INHERITED_SCORES_COLUMN, PLAYER_INNINGS_PITCHED_COLUMN,
    PLAYER_INSIDES_COLUMN, PLAYER_INSWINGING_CORNER_KICKS_COLUMN,
    PLAYER_INTERCEPTION_PERCENTAGE_COLUMN,
    PLAYER_INTERCEPTION_TOUCHDOWNS_COLUMN, PLAYER_INTERCEPTION_YARDS_COLUMN,
    PLAYER_INTERCEPTIONS_COLUMN, PLAYER_KEY_PASSES_COLUMN,
    PLAYER_KICK_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_KICK_RETURN_FAIR_CATCHES_COLUMN, PLAYER_KICK_RETURN_FUMBLES_COLUMN,
    PLAYER_KICK_RETURN_FUMBLES_LOST_COLUMN,
    PLAYER_KICK_RETURN_TOUCHDOWNS_COLUMN, PLAYER_KICK_RETURN_YARDS_COLUMN,
    PLAYER_KICK_RETURNS_COLUMN, PLAYER_KICKOFF_OUT_OF_BOUNDS_COLUMN,
    PLAYER_KICKOFF_RETURN_YARDS_COLUMN, PLAYER_KICKOFF_RETURNS_COLUMN,
    PLAYER_KICKOFF_RETURNS_TOUCHDOWNS_COLUMN, PLAYER_KICKOFF_YARDS_COLUMN,
    PLAYER_KICKOFFS_COLUMN, PLAYER_KICKS_BLOCKED_COLUMN, PLAYER_KICKS_COLUMN,
    PLAYER_LINE_DRIVES_COLUMN, PLAYER_LIVE_BALL_PASSES_COLUMN,
    PLAYER_LIVE_BALL_TOUCHES_COLUMN, PLAYER_LONG_FIELD_GOAL_ATTEMPT_COLUMN,
    PLAYER_LONG_FIELD_GOAL_MADE_COLUMN, PLAYER_LONG_INTERCEPTION_COLUMN,
    PLAYER_LONG_KICK_RETURN_COLUMN, PLAYER_LONG_KICKOFF_COLUMN,
    PLAYER_LONG_PASSING_COLUMN, PLAYER_LONG_PUNT_COLUMN,
    PLAYER_LONG_PUNT_RETURN_COLUMN, PLAYER_LONG_RECEPTION_COLUMN,
    PLAYER_LONG_RUSHING_COLUMN, PLAYER_MAIDENS_COLUMN, PLAYER_MARKS_COLUMN,
    PLAYER_MARKS_INSIDE_COLUMN, PLAYER_MISC_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_MISC_FUMBLE_RETURNS_COLUMN, PLAYER_MISC_POINTS_COLUMN,
    PLAYER_MISC_TOUCHDOWNS_COLUMN, PLAYER_MISC_YARDS_COLUMN,
    PLAYER_MISCONTROLS_COLUMN, PLAYER_MISSED_FIELD_GOAL_RETURN_TD_COLUMN,
    PLAYER_NET_AVERAGE_PUNT_YARDS_COLUMN, PLAYER_NET_PASSING_ATTEMPTS_COLUMN,
    PLAYER_NET_PASSING_YARDS_COLUMN, PLAYER_NET_TOTAL_YARDS_COLUMN,
    PLAYER_NET_YARDS_PER_PASS_ATTEMPT_COLUMN, PLAYER_NO_BALLS_COLUMN,
    PLAYER_NON_PENALTY_EXPECTED_GOALS_COLUMN, PLAYER_OBP_COLUMN,
    PLAYER_OFFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN, PLAYER_OFFENSIVE_RATING_COLUMN,
    PLAYER_OFFENSIVE_REBOUND_PERCENTAGE_COLUMN,
    PLAYER_OFFENSIVE_TWO_POINT_RETURNS_COLUMN,
    PLAYER_OFFENSIVE_ZONE_START_PERCENTAGE_COLUMN,
    PLAYER_OFFENSIVE_ZONE_STARTS_COLUMN, PLAYER_OFFSIDES_COLUMN,
    PLAYER_ON_SHOT_ICE_AGAINST_EVENTS_COLUMN,
    PLAYER_ON_SHOT_ICE_FOR_EVENTS_COLUMN, PLAYER_ONE_PERCENTERS_COLUMN,
    PLAYER_ONE_POINT_SAFETIES_MADE_COLUMN,
    PLAYER_OPPOSITION_FUMBLE_RECOVERIES_COLUMN,
    PLAYER_OPPOSITION_FUMBLE_RECOVERY_YARDS_COLUMN,
    PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN, PLAYER_OPS_COLUMN,
    PLAYER_OUTSWINGING_CORNER_KICKS_COLUMN, PLAYER_OVERS_COLUMN,
    PLAYER_OWN_GOALS_COLUMN, PLAYER_PASS_COMPLETION_COLUMN,
    PLAYER_PASS_COMPLETION_LONG_COLUMN, PLAYER_PASS_COMPLETION_MEDIUM_COLUMN,
    PLAYER_PASS_COMPLETION_SHORT_COLUMN, PLAYER_PASSES_ATTEMPTED_COLUMN,
    PLAYER_PASSES_ATTEMPTED_LONG_COLUMN, PLAYER_PASSES_ATTEMPTED_MEDIUM_COLUMN,
    PLAYER_PASSES_ATTEMPTED_MINUS_GOAL_KICKS_COLUMN,
    PLAYER_PASSES_ATTEMPTED_SHORT_COLUMN, PLAYER_PASSES_BATTED_DOWN_COLUMN,
    PLAYER_PASSES_BLOCKED_COLUMN, PLAYER_PASSES_COMPLETED_COLUMN,
    PLAYER_PASSES_COMPLETED_LONG_COLUMN, PLAYER_PASSES_COMPLETED_MEDIUM_COLUMN,
    PLAYER_PASSES_COMPLETED_SHORT_COLUMN, PLAYER_PASSES_DEFENDED_COLUMN,
    PLAYER_PASSES_FROM_FREE_KICKS_COLUMN,
    PLAYER_PASSES_INTO_FINAL_THIRD_COLUMN,
    PLAYER_PASSES_INTO_PENALTY_AREA_COLUMN, PLAYER_PASSES_OFFSIDE_COLUMN,
    PLAYER_PASSES_RECEIVED_COLUMN, PLAYER_PASSING_ATTEMPTS_COLUMN,
    PLAYER_PASSING_BIG_PLAYS_COLUMN, PLAYER_PASSING_FIRST_DOWNS_COLUMN,
    PLAYER_PASSING_FUMBLES_COLUMN, PLAYER_PASSING_FUMBLES_LOST_COLUMN,
    PLAYER_PASSING_TOUCHDOWN_PERCENTAGE_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_PASSING_TOUCHDOWNS_COLUMN, PLAYER_PASSING_YARDS_AFTER_CATCH_COLUMN,
    PLAYER_PASSING_YARDS_AT_CATCH_COLUMN, PLAYER_PASSING_YARDS_COLUMN,
    PLAYER_PENALTIES_IN_MINUTES_COLUMN, PLAYER_PENALTY_KICKS_ATTEMPTED_COLUMN,
    PLAYER_PENALTY_KICKS_CONCEDED_COLUMN, PLAYER_PENALTY_KICKS_MADE_COLUMN,
    PLAYER_PENALTY_KICKS_WON_COLUMN,
    PLAYER_PERCENT_OF_DRIBBLERS_TACKLED_COLUMN,
    PLAYER_PERCENTAGE_CROSSES_STOPPED_COLUMN,
    PLAYER_PERCENTAGE_OF_AERIALS_WON_COLUMN,
    PLAYER_PERCENTAGE_OF_GOAL_KICKS_THAT_WERE_LAUNCHED_COLUMN,
    PLAYER_PERCENTAGE_OF_PASSES_THAT_WERE_LAUNCHED_COLUMN,
    PLAYER_PERCENTAGE_PLAYED_COLUMN, PLAYER_PERSONAL_FOULS_COLUMN,
    PLAYER_PITCHES_COLUMN, PLAYER_PLATE_APPEARANCES_COLUMN,
    PLAYER_PLAYING_ROLES_COLUMN, PLAYER_POINT_DIFFERENTIAL_COLUMN,
    PLAYER_POINTS_ALLOWED_COLUMN, PLAYER_POINTS_COLUMN,
    PLAYER_POST_SHOT_EXPECTED_GOALS_COLUMN, PLAYER_POWER_PLAY_ASSISTS_COLUMN,
    PLAYER_POWER_PLAY_GOALS_COLUMN, PLAYER_PROGRESSIVE_CARRIES_COLUMN,
    PLAYER_PROGRESSIVE_CARRYING_DISTANCE_COLUMN,
    PLAYER_PROGRESSIVE_PASSES_COLUMN,
    PLAYER_PROGRESSIVE_PASSES_RECEIVED_COLUMN,
    PLAYER_PROGRESSIVE_PASSING_DISTANCE_COLUMN,
    PLAYER_PUNT_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
    PLAYER_PUNT_RETURN_FAIR_CATCHES_COLUMN, PLAYER_PUNT_RETURN_FUMBLES_COLUMN,
    PLAYER_PUNT_RETURN_FUMBLES_LOST_COLUMN,
    PLAYER_PUNT_RETURN_TOUCHDOWNS_COLUMN, PLAYER_PUNT_RETURN_YARDS_COLUMN,
    PLAYER_PUNT_RETURNS_COLUMN,
    PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_10_COLUMN,
    PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_20_COLUMN, PLAYER_PUNT_YARDS_COLUMN,
    PLAYER_PUNTS_BLOCKED_COLUMN, PLAYER_PUNTS_BLOCKED_PERCENTAGE_COLUMN,
    PLAYER_PUNTS_COLUMN, PLAYER_PUNTS_INSIDE_10_COLUMN,
    PLAYER_PUNTS_INSIDE_10_PERCENTAGE_COLUMN, PLAYER_PUNTS_INSIDE_20_COLUMN,
    PLAYER_PUNTS_INSIDE_20_PERCENTAGE_COLUMN, PLAYER_PUNTS_OVER_50_COLUMN,
    PLAYER_PUTOUTS_COLUMN, PLAYER_QUARTERBACK_HITS_COLUMN,
    PLAYER_QUARTERBACK_RATING_COLUMN, PLAYER_RE24_COLUMN,
    PLAYER_REBOUNDS_COLUMN, PLAYER_RECEIVING_BIG_PLAYS_COLUMN,
    PLAYER_RECEIVING_FIRST_DOWNS_COLUMN, PLAYER_RECEIVING_FUMBLES_COLUMN,
    PLAYER_RECEIVING_FUMBLES_LOST_COLUMN, PLAYER_RECEIVING_TARGETS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_RECEIVING_TOUCHDOWNS_COLUMN,
    PLAYER_RECEIVING_YARDS_AFTER_CATCH_COLUMN,
    PLAYER_RECEIVING_YARDS_AT_CATCH_COLUMN, PLAYER_RECEIVING_YARDS_COLUMN,
    PLAYER_RECEPTIONS_COLUMN, PLAYER_RED_CARDS_COLUMN,
    PLAYER_RELATIVE_CORSI_FOR_PERCENTAGE_COLUMN,
    PLAYER_RETURN_TOUCHDOWNS_COLUMN, PLAYER_RUNS_BATTED_IN_COLUMN,
    PLAYER_RUNS_COLUMN, PLAYER_RUNS_PER_BALL_COLUMN, PLAYER_RUNS_SCORED_COLUMN,
    PLAYER_RUSHING_ATTEMPTS_COLUMN, PLAYER_RUSHING_BIG_PLAYS_COLUMN,
    PLAYER_RUSHING_FIRST_DOWNS_COLUMN, PLAYER_RUSHING_FUMBLES_COLUMN,
    PLAYER_RUSHING_FUMBLES_LOST_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_9_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_19_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_29_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_39_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_49_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
    PLAYER_RUSHING_TOUCHDOWNS_COLUMN, PLAYER_RUSHING_YARDS_COLUMN,
    PLAYER_SACKS_ASSISTED_COLUMN, PLAYER_SACKS_COLUMN,
    PLAYER_SACKS_UNASSISTED_COLUMN, PLAYER_SACKS_YARDS_COLUMN,
    PLAYER_SACKS_YARDS_LOST_COLUMN, PLAYER_SAFETIES_COLUMN,
    PLAYER_SAVE_PERCENTAGE_COLUMN, PLAYER_SAVES_COLUMN,
    PLAYER_SECOND_YELLOW_CARD_COLUMN, PLAYER_SECONDS_PLAYED_COLUMN,
    PLAYER_SHIFTS_COLUMN, PLAYER_SHOOTING_PERCENTAGE_COLUMN,
    PLAYER_SHORT_HANDED_ASSISTS_COLUMN, PLAYER_SHORT_HANDED_GOALS_COLUMN,
    PLAYER_SHOT_CREATING_ACTIONS_COLUMN, PLAYER_SHOTS_AGAINST_COLUMN,
    PLAYER_SHOTS_BLOCKED_COLUMN, PLAYER_SHOTS_ON_GOAL_COLUMN,
    PLAYER_SHOTS_ON_TARGET_AGAINST_COLUMN, PLAYER_SHOTS_ON_TARGET_COLUMN,
    PLAYER_SHOTS_TOTAL_COLUMN, PLAYER_SHUTOUTS_COLUMN, PLAYER_SIXES_COLUMN,
    PLAYER_SLG_COLUMN, PLAYER_SOLO_TACKLES_COLUMN,
    PLAYER_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
    PLAYER_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN, PLAYER_STEAL_PERCENTAGE_COLUMN,
    PLAYER_STEALS_COLUMN, PLAYER_STRAIGHT_CORNER_KICKS_COLUMN,
    PLAYER_STRIKEOUTS_COLUMN, PLAYER_STRIKERATE_COLUMN,
    PLAYER_STRIKES_BY_CONTACT_COLUMN, PLAYER_STRIKES_COLUMN,
    PLAYER_STRIKES_LOOKING_COLUMN, PLAYER_STRIKES_SWINGING_COLUMN,
    PLAYER_STUFF_YARDS_COLUMN, PLAYER_STUFF_YARDS_LOST, PLAYER_STUFFS_COLUMN,
    PLAYER_SUCCESSFUL_TAKE_ON_PERCENTAGE_COLUMN,
    PLAYER_SUCCESSFUL_TAKE_ONS_COLUMN, PLAYER_SWITCHES_COLUNM,
    PLAYER_TACKLED_DURING_TAKE_ON_PERCENTAGE_COLUMN, PLAYER_TACKLES_COLUMN,
    PLAYER_TACKLES_FOR_LOSS_COLUMN, PLAYER_TACKLES_IN_ATTACKING_THIRD_COLUMN,
    PLAYER_TACKLES_IN_DEFENSIVE_THIRD_COLUMN,
    PLAYER_TACKLES_IN_MIDDLE_THIRD_COLUMN,
    PLAYER_TACKLES_PLUS_INTERCEPTIONS_COLUMN, PLAYER_TACKLES_WON_COLUMN,
    PLAYER_TACKLES_YARDS_LOST_COLUMN, PLAYER_TAKE_ONS_ATTEMPTED_COLUMN,
    PLAYER_THREE_POINT_ATTEMPT_RATE_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_COLUMN,
    PLAYER_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
    PLAYER_THROUGH_BALLS_COLUMN, PLAYER_THROW_INS_TAKEN_COLUMN,
    PLAYER_THROWS_ATTEMPTED_COLUMN, PLAYER_TIME_ON_ICE_COLUMN,
    PLAYER_TIMES_TACKLED_DURING_TAKE_ONS_COLUMN,
    PLAYER_TOTAL_CARRYING_DISTANCE_COLUMN, PLAYER_TOTAL_KICKING_POINTS_COLUMN,
    PLAYER_TOTAL_OFFENSIVE_PLAYS_COLUMN, PLAYER_TOTAL_PASSING_DISTANCE_COLUMN,
    PLAYER_TOTAL_POINTS_COLUMN, PLAYER_TOTAL_REBOUND_PERCENTAGE_COLUMN,
    PLAYER_TOTAL_REBOUNDS_COLUMN, PLAYER_TOTAL_TOUCHDOWNS_COLUMN,
    PLAYER_TOTAL_TWO_POINT_CONVERSIONS_COLUMN, PLAYER_TOTAL_YARDS_COLUMN,
    PLAYER_TOTAL_YARDS_FROM_SCRIMMAGE_COLUMN,
    PLAYER_TOUCHBACK_PERCENTAGE_COLUMN, PLAYER_TOUCHBACKS_COLUMN,
    PLAYER_TOUCHES_COLUMN, PLAYER_TOUCHES_IN_ATTACKING_PENALTY_AREA_COLUMN,
    PLAYER_TOUCHES_IN_ATTACKING_THIRD_COLUMN,
    PLAYER_TOUCHES_IN_DEFENSIVE_PENALTY_AREA_COLUMN,
    PLAYER_TOUCHES_IN_DEFENSIVE_THIRD_COLUMN,
    PLAYER_TOUCHES_IN_MIDDLE_THIRD_COLUMN,
    PLAYER_TRUE_SHOOTING_PERCENTAGE_COLUMN, PLAYER_TURNOVER_PERCENTAGE_COLUMN,
    PLAYER_TWO_POINT_PASS_ATTEMPT_COLUMN, PLAYER_TWO_POINT_PASS_COLUMN,
    PLAYER_TWO_POINT_RECEPTION_ATTEMPTS_COLUMN,
    PLAYER_TWO_POINT_RECEPTIONS_COLUMN, PLAYER_TWO_POINT_RUSH_ATTEMPTS_COLUMN,
    PLAYER_TWO_POINT_RUSH_COLUMN, PLAYER_UNCONTESTED_POSSESSIONS_COLUMN,
    PLAYER_USAGE_PERCENTAGE_COLUMN, PLAYER_WICKETS_COLUMN, PLAYER_WIDES_COLUMN,
    PLAYER_WIN_PROBABILITY_ADDED_COLUMN, PLAYER_WPA_MINUS_COLUMN,
    PLAYER_WPA_PLUS_COLUMN, PLAYER_YARDS_ALLOWED_COLUMN,
    PLAYER_YARDS_PER_COMPLETION_COLUMN, PLAYER_YARDS_PER_KICK_RETURN_COLUMN,
    PLAYER_YARDS_PER_PASS_ATTEMPT_COLUMN, PLAYER_YARDS_PER_RECEPTION_COLUMN,
    PLAYER_YARDS_PER_RETURN_COLUMN, PLAYER_YARDS_PER_RUSH_ATTEMPT_COLUMN,
    PLAYER_YELLOW_CARDS_COLUMN)
from sportsball.data.player_model import \
    TURNOVERS_COLUMN as PLAYER_TURNOVERS_COLUMN  # type: ignore
from sportsball.data.team_model import ASSISTS_COLUMN  # type: ignore
from sportsball.data.team_model import (
    FIELD_GOALS_ATTEMPTED_COLUMN, FIELD_GOALS_COLUMN, KICKS_COLUMN,
    OFFENSIVE_REBOUNDS_COLUMN, TEAM_BALLS_COLUMN, TEAM_BALLS_PER_OVER_COLUMN,
    TEAM_BEHINDS_COLUMN, TEAM_BLOCKS_COLUMN, TEAM_BOUNCES_COLUMN,
    TEAM_BROWNLOW_VOTES_COLUMN, TEAM_BYES_COLUMN, TEAM_CATCHES_COLUMN,
    TEAM_CATCHES_DROPPED_COLUMN, TEAM_CLANGERS_COLUMN, TEAM_CLEARANCES_COLUMN,
    TEAM_CONTESTED_MARKS_COLUMN, TEAM_CONTESTED_POSSESSIONS_COLUMN,
    TEAM_DEFENSIVE_REBOUNDS_COLUMN, TEAM_DISPOSALS_COLUMN,
    TEAM_FIELD_GOALS_PERCENTAGE_COLUMN, TEAM_FORCED_FUMBLES_COLUMN,
    TEAM_FOURS_COLUMN, TEAM_FREE_KICKS_AGAINST_COLUMN,
    TEAM_FREE_KICKS_FOR_COLUMN, TEAM_FREE_THROWS_ATTEMPTED_COLUMN,
    TEAM_FREE_THROWS_COLUMN, TEAM_FREE_THROWS_PERCENTAGE_COLUMN,
    TEAM_FUMBLES_RECOVERED_COLUMN, TEAM_FUMBLES_TOUCHDOWNS_COLUMN,
    TEAM_GOAL_ASSISTS_COLUMN, TEAM_GOALS_COLUMN, TEAM_HANDBALLS_COLUMN,
    TEAM_HIT_OUTS_COLUMN, TEAM_INSIDES_COLUMN, TEAM_LEG_BYES_COLUMN,
    TEAM_LENGTH_BEHIND_WINNER_COLUMN, TEAM_MARKS_COLUMN,
    TEAM_MARKS_INSIDE_COLUMN, TEAM_NO_BALLS_COLUMN, TEAM_ONE_PERCENTERS_COLUMN,
    TEAM_OVERS_COLUMN, TEAM_PENALTIES_COLUMN, TEAM_PERSONAL_FOULS_COLUMN,
    TEAM_REBOUNDS_COLUMN, TEAM_RUNS_COLUMN, TEAM_SIXES_COLUMN,
    TEAM_STEALS_COLUMN, TEAM_TACKLES_COLUMN,
    TEAM_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
    TEAM_THREE_POINT_FIELD_GOALS_COLUMN,
    TEAM_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN, TEAM_TOTAL_REBOUNDS_COLUMN,
    TEAM_UNCONTESTED_POSSESSIONS_COLUMN, TEAM_WICKETS_COLUMN,
    TEAM_WIDES_COLUMN, TURNOVERS_COLUMN)
from sportsball.data.venue_model import VENUE_ADDRESS_COLUMN
from sportsfeatures.bet import Bet
from sportsfeatures.embedding_column import is_embedding_column
from sportsfeatures.entity_type import EntityType  # type: ignore
from sportsfeatures.identifier import Identifier  # type: ignore
from sportsfeatures.news import News
from sportsfeatures.process import process  # type: ignore

from .features.columns import (coach_column_prefix, coach_identifier_column,
                               find_coach_count, find_news_count,
                               find_odds_count, find_player_count,
                               find_team_count, news_column_prefix,
                               news_summary_column, odds_column_prefix,
                               odds_odds_column, player_column_prefix,
                               player_identifier_column, team_column_prefix,
                               team_identifier_column, team_name_column,
                               team_points_column, venue_identifier_column)
from .kelly_fractions import (augment_kelly_fractions, calculate_returns,
                              calculate_value)

AWAY_WIN_COLUMN = "away_win"

_DF_FILENAME = "df.parquet.gzip"
_CONFIG_FILENAME = "config.json"
_PLACE_KEY = "place"
_VALIDATION_SIZE = datetime.timedelta(days=365)
_TEST_SIZE = datetime.timedelta(days=365)
_SAMPLER_FILENAME = "sampler.pkl"
_KELLY_KEY = "kelly"
_ALPHA_KEY = "alpha"


class Strategy:
    """The strategy class."""

    # pylint: disable=too-many-locals,too-many-instance-attributes

    _returns: pd.Series | None
    _place: int

    def __init__(self, name: str, place: int | None = None) -> None:
        self._df = None
        self._name = name
        os.makedirs(name, exist_ok=True)

        # Load dataframe previously used.
        df_file = os.path.join(name, _DF_FILENAME)
        if os.path.exists(df_file):
            self._df = pd.read_parquet(df_file)

        self._wt = wt.create(
            self._name,
            dt_column=GAME_DT_COLUMN,
            walkforward_timedelta=datetime.timedelta(days=7),
            validation_size=_VALIDATION_SIZE,
            max_train_timeout=datetime.timedelta(hours=12),
            cutoff_dt=datetime.datetime.now(tz=pytz.UTC),
            test_size=_TEST_SIZE,
            allowed_models={"catboost"},
            max_false_positive_reduction_steps=1,
            correlation_chunk_size=5000,
            insert_null=True,
        )

        # Load config
        config_filename = os.path.join(name, _CONFIG_FILENAME)
        if os.path.exists(config_filename) and place is None:
            with open(config_filename, "r", encoding="utf8") as handle:
                config = json.load(handle)
                place = config.get(_PLACE_KEY)
        elif place is not None:
            with open(config_filename, "w", encoding="utf8") as handle:
                json.dump({_PLACE_KEY: place}, handle)
        self._place = place if place is not None else 1

        self._returns = None

        storage_name = f"sqlite:///{name}/study.db"
        sampler_file = os.path.join(name, _SAMPLER_FILENAME)
        restored_sampler = None
        if os.path.exists(sampler_file):
            with open(sampler_file, "rb") as handle:
                restored_sampler = pickle.load(handle)
        self._study = optuna.create_study(
            study_name=name,
            storage=storage_name,
            load_if_exists=True,
            sampler=restored_sampler,
            direction=optuna.study.StudyDirection.MAXIMIZE,
        )

    @property
    def df(self) -> pd.DataFrame | None:
        """Fetch the dataframe currently being operated on."""
        df = self._df
        if df is None:
            return None
        return df.sort_values(by=DT_COLUMN, ascending=True)

    @df.setter
    def df(self, df: pd.DataFrame) -> None:
        """Set the dataframe."""
        self._df = df.sort_values(by=DT_COLUMN, ascending=True)
        df.to_parquet(os.path.join(self._name, _DF_FILENAME), compression="gzip")
        self._df = pd.read_parquet(os.path.join(self._name, _DF_FILENAME))

    @property
    def name(self) -> str:
        """Fetch the name of the strategy."""
        return self._name

    def find_returns(self, df: pd.DataFrame) -> pd.Series:
        """Find the best kelly ratio for this strategy."""
        main_df = self.df
        if main_df is None:
            raise ValueError("main_df is null")
        points_cols = main_df.attrs[str(FieldType.POINTS)]
        df[points_cols] = main_df[points_cols].to_numpy()
        cutoff_dt = pd.to_datetime(datetime.datetime.now() - _VALIDATION_SIZE).date()
        df = df[df[GAME_DT_COLUMN].dt.date > cutoff_dt]

        def trial_returns(
            trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
        ) -> pd.Series:
            alpha = trial.suggest_float(_ALPHA_KEY, 0.0, 2.0)
            kelly_threshold = trial.suggest_float(_KELLY_KEY, 0.0, 1.0)

            df = augment_kelly_fractions(df, len(points_cols), alpha)
            returns = calculate_returns(
                kelly_threshold,
                df,
                self._name,
            )
            return returns

        def run_trial(
            trial: optuna.Trial | optuna.trial.FrozenTrial, df: pd.DataFrame
        ) -> float:
            returns = trial_returns(trial, df)
            value = calculate_value(returns)
            return value

        if not self._study.trials:
            self._study.optimize(
                functools.partial(
                    run_trial,
                    df=df,
                ),
                n_trials=100,
                timeout=60.0 * 60.0 * 5,
                show_progress_bar=True,
            )

        return trial_returns(self._study.best_trial, df)

    def fit(self):
        """Fits the strategy to the dataset by walking forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null")
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = self._process()
        y = df[training_cols]
        teams = find_team_count(df)

        def make_y() -> pd.Series | pd.DataFrame:
            nonlocal y
            if teams == 2:
                y_max = np.argmax(y.to_numpy(), axis=1)
                y[AWAY_WIN_COLUMN] = y_max
                y[AWAY_WIN_COLUMN] = y[AWAY_WIN_COLUMN].astype(bool)
                return y[AWAY_WIN_COLUMN]
            ind = np.argpartition(y.to_numpy(), -self._place)[-self._place :]
            for i in range(teams):
                y[DELIMITER.join(["team", str(i), "win"])] = i in ind
            return y.drop(columns=training_cols)  # type: ignore

        y = make_y()
        x_df = x_df.drop(columns=training_cols)
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        self._wt.embedding_cols = self._calculate_embedding_columns(x_df)
        self._wt.fit(x_df, y=y)

    def predict(self) -> pd.DataFrame:
        """Predict the results from walk-forward."""
        df = self.df
        if df is None:
            raise ValueError("df is null.")

        x_df = self._process()
        training_cols = df.attrs[str(FieldType.POINTS)]
        x_df = x_df.drop(columns=training_cols, errors="ignore")
        x_df = x_df.drop(columns=df.attrs[str(FieldType.LOOKAHEAD)], errors="ignore")
        self._wt.embedding_cols = self._calculate_embedding_columns(x_df)

        # Ensure correct odds
        today = (datetime.datetime.today() - datetime.timedelta(days=1)).date()
        future_rows = x_df[x_df[GAME_DT_COLUMN].dt.date >= today]
        for idx, row in future_rows.iterrows():
            for team_id in range(find_team_count(x_df)):
                odds_col = f"teams/{team_id}_odds"
                if pd.isna(row.get(odds_col)):
                    while True:
                        name_col = team_name_column(team_id)
                        try:
                            new_odds = float(
                                input(
                                    f"Enter new odds for {odds_col} at row {idx} for team {row.get(name_col)} @ {row.get(GAME_DT_COLUMN)}: "
                                )
                            )
                            x_df.at[idx, odds_col] = new_odds
                            break
                        except ValueError:
                            print("Invalid input. Please enter a numeric value.")

        x_df = self._wt.transform(x_df)
        for points_col in df.attrs[str(FieldType.POINTS)]:
            x_df[points_col] = df[points_col]
        return x_df

    def returns(self) -> pd.Series:
        """Render the returns of the strategy."""
        df = self.predict()
        returns = self.find_returns(df)
        if returns is None:
            raise ValueError("returns is null")
        return returns

    def next(
        self,
    ) -> tuple[
        pd.DataFrame,
        dict[str, tuple[dict[str, float], list[dict[str, float]]]],
        float,
        float,
    ]:
        """Find the next predictions for betting."""
        dt_column = DELIMITER.join([GAME_DT_COLUMN])
        df = self.predict()
        self.find_returns(df)
        kelly_ratio = self._study.best_trial.suggest_float(_KELLY_KEY, 0.0, 1.0)
        alpha = self._study.best_trial.suggest_float(_ALPHA_KEY, 0.0, 2.0)
        start_dt = datetime.datetime.now(datetime.timezone.utc)
        end_dt = start_dt + datetime.timedelta(days=3.0)
        df = df[df[dt_column] > start_dt]
        df = df[df[dt_column] <= end_dt]
        return (
            df,
            self._wt.feature_importances(df=df, latest_date_only=True),
            kelly_ratio,
            alpha,
        )

    def _process(self) -> pd.DataFrame:
        df = self.df
        if df is None:
            raise ValueError("df is null")

        df_hash = hashlib.sha256(df.to_csv().encode()).hexdigest()
        df_cache_path = os.path.join(self._name, f"processed_{df_hash}.parquet")
        if os.path.exists(df_cache_path):
            return pd.read_parquet(df_cache_path)

        team_count = find_team_count(df)

        identifiers = [
            Identifier(
                EntityType.VENUE,
                venue_identifier_column(),
                [],
                VENUE_COLUMN_PREFIX,
                latitude_column=DELIMITER.join(
                    [VENUE_COLUMN_PREFIX, VENUE_ADDRESS_COLUMN, ADDRESS_LATITUDE_COLUMN]
                ),
                longitude_column=DELIMITER.join(
                    [
                        VENUE_COLUMN_PREFIX,
                        VENUE_ADDRESS_COLUMN,
                        ADDRESS_LONGITUDE_COLUMN,
                    ]
                ),
            )
        ]
        odds_count = find_odds_count(df, team_count)
        news_count = find_news_count(df, team_count)
        datetime_columns: set[str] = set()
        for i in range(team_count):
            identifiers.append(
                Identifier(
                    EntityType.TEAM,
                    team_identifier_column(i),
                    [
                        DELIMITER.join([team_column_prefix(i), x])
                        for x in [
                            FIELD_GOALS_COLUMN,
                            FIELD_GOALS_ATTEMPTED_COLUMN,
                            OFFENSIVE_REBOUNDS_COLUMN,
                            ASSISTS_COLUMN,
                            TURNOVERS_COLUMN,
                            KICKS_COLUMN,
                            TEAM_MARKS_COLUMN,
                            TEAM_HANDBALLS_COLUMN,
                            TEAM_DISPOSALS_COLUMN,
                            TEAM_GOALS_COLUMN,
                            TEAM_BEHINDS_COLUMN,
                            TEAM_HIT_OUTS_COLUMN,
                            TEAM_TACKLES_COLUMN,
                            TEAM_REBOUNDS_COLUMN,
                            TEAM_INSIDES_COLUMN,
                            TEAM_CLEARANCES_COLUMN,
                            TEAM_CLANGERS_COLUMN,
                            TEAM_FREE_KICKS_FOR_COLUMN,
                            TEAM_FREE_KICKS_AGAINST_COLUMN,
                            TEAM_BROWNLOW_VOTES_COLUMN,
                            TEAM_CONTESTED_POSSESSIONS_COLUMN,
                            TEAM_UNCONTESTED_POSSESSIONS_COLUMN,
                            TEAM_CONTESTED_MARKS_COLUMN,
                            TEAM_MARKS_INSIDE_COLUMN,
                            TEAM_ONE_PERCENTERS_COLUMN,
                            TEAM_BOUNCES_COLUMN,
                            TEAM_GOAL_ASSISTS_COLUMN,
                            TEAM_LENGTH_BEHIND_WINNER_COLUMN,
                            TEAM_FIELD_GOALS_PERCENTAGE_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
                            TEAM_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
                            TEAM_FREE_THROWS_COLUMN,
                            TEAM_FREE_THROWS_ATTEMPTED_COLUMN,
                            TEAM_FREE_THROWS_PERCENTAGE_COLUMN,
                            TEAM_DEFENSIVE_REBOUNDS_COLUMN,
                            TEAM_TOTAL_REBOUNDS_COLUMN,
                            TEAM_STEALS_COLUMN,
                            TEAM_BLOCKS_COLUMN,
                            TEAM_PERSONAL_FOULS_COLUMN,
                            TEAM_FORCED_FUMBLES_COLUMN,
                            TEAM_FUMBLES_RECOVERED_COLUMN,
                            TEAM_FUMBLES_TOUCHDOWNS_COLUMN,
                            TEAM_RUNS_COLUMN,
                            TEAM_WICKETS_COLUMN,
                            TEAM_OVERS_COLUMN,
                            TEAM_BALLS_COLUMN,
                            TEAM_BYES_COLUMN,
                            TEAM_LEG_BYES_COLUMN,
                            TEAM_WIDES_COLUMN,
                            TEAM_NO_BALLS_COLUMN,
                            TEAM_PENALTIES_COLUMN,
                            TEAM_BALLS_PER_OVER_COLUMN,
                            TEAM_FOURS_COLUMN,
                            TEAM_SIXES_COLUMN,
                            TEAM_CATCHES_COLUMN,
                            TEAM_CATCHES_DROPPED_COLUMN,
                        ]
                    ],
                    team_column_prefix(i),
                    points_column=team_points_column(i),
                    field_goals_column=DELIMITER.join(
                        [team_column_prefix(i), FIELD_GOALS_COLUMN]
                    ),
                    assists_column=DELIMITER.join(
                        [team_column_prefix(i), ASSISTS_COLUMN]
                    ),
                    field_goals_attempted_column=DELIMITER.join(
                        [team_column_prefix(i), FIELD_GOALS_ATTEMPTED_COLUMN]
                    ),
                    offensive_rebounds_column=DELIMITER.join(
                        [team_column_prefix(i), OFFENSIVE_REBOUNDS_COLUMN]
                    ),
                    turnovers_column=DELIMITER.join(
                        [team_column_prefix(i), TURNOVERS_COLUMN]
                    ),
                    bets=[
                        Bet(
                            odds_column=odds_odds_column(i, x),
                            bookie_id_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BOOKIE_COLUMN,
                                    BOOKIE_IDENTIFIER_COLUMN,
                                ]
                            ),
                            dt_column=DELIMITER.join(
                                [odds_column_prefix(i, x), DT_COLUMN]
                            ),
                            canonical_column=DELIMITER.join(
                                [odds_column_prefix(i, x), ODDS_CANONICAL_COLUMN]
                            ),
                            bookie_name_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BOOKIE_COLUMN,
                                    "name",
                                ]
                            ),
                            bet_type_column=DELIMITER.join(
                                [
                                    odds_column_prefix(i, x),
                                    ODDS_BET_COLUMN,
                                ]
                            ),
                        )
                        for x in range(odds_count)
                    ],
                    news=[
                        News(
                            title_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_TITLE_COLUMN]
                            ),
                            published_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_PUBLISHED_COLUMN]
                            ),
                            summary_column=news_summary_column(i, x),
                            source_column=DELIMITER.join(
                                [news_column_prefix(i, x), NEWS_SOURCE_COLUMN]
                            ),
                        )
                        for x in range(news_count)
                    ],
                )
            )
            player_count = find_player_count(df, i)
            identifiers.extend(
                [
                    Identifier(
                        EntityType.PLAYER,
                        player_identifier_column(i, x),
                        [
                            DELIMITER.join([player_column_prefix(i, x), col])
                            for col in [
                                PLAYER_KICKS_COLUMN,
                                PLAYER_FUMBLES_COLUMN,
                                PLAYER_FUMBLES_LOST_COLUMN,
                                PLAYER_FIELD_GOALS_COLUMN,
                                PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN,
                                PLAYER_OFFENSIVE_REBOUNDS_COLUMN,
                                PLAYER_ASSISTS_COLUMN,
                                PLAYER_TURNOVERS_COLUMN,
                                PLAYER_MARKS_COLUMN,
                                PLAYER_HANDBALLS_COLUMN,
                                PLAYER_DISPOSALS_COLUMN,
                                PLAYER_GOALS_COLUMN,
                                PLAYER_BEHINDS_COLUMN,
                                PLAYER_HIT_OUTS_COLUMN,
                                PLAYER_TACKLES_COLUMN,
                                PLAYER_REBOUNDS_COLUMN,
                                PLAYER_INSIDES_COLUMN,
                                PLAYER_CLEARANCES_COLUMN,
                                PLAYER_CLANGERS_COLUMN,
                                PLAYER_FREE_KICKS_FOR_COLUMN,
                                PLAYER_FREE_KICKS_AGAINST_COLUMN,
                                PLAYER_BROWNLOW_VOTES_COLUMN,
                                PLAYER_CONTESTED_POSSESSIONS_COLUMN,
                                PLAYER_UNCONTESTED_POSSESSIONS_COLUMN,
                                PLAYER_CONTESTED_MARKS_COLUMN,
                                PLAYER_MARKS_INSIDE_COLUMN,
                                PLAYER_ONE_PERCENTERS_COLUMN,
                                PLAYER_BOUNCES_COLUMN,
                                PLAYER_GOAL_ASSISTS_COLUMN,
                                PLAYER_PERCENTAGE_PLAYED_COLUMN,
                                PLAYER_SECONDS_PLAYED_COLUMN,
                                PLAYER_FIELD_GOALS_PERCENTAGE_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_ATTEMPTED_COLUMN,
                                PLAYER_THREE_POINT_FIELD_GOALS_PERCENTAGE_COLUMN,
                                PLAYER_FREE_THROWS_COLUMN,
                                PLAYER_FREE_THROWS_ATTEMPTED_COLUMN,
                                PLAYER_FREE_THROWS_PERCENTAGE_COLUMN,
                                PLAYER_DEFENSIVE_REBOUNDS_COLUMN,
                                PLAYER_TOTAL_REBOUNDS_COLUMN,
                                PLAYER_STEALS_COLUMN,
                                PLAYER_BLOCKS_COLUMN,
                                PLAYER_PERSONAL_FOULS_COLUMN,
                                PLAYER_POINTS_COLUMN,
                                PLAYER_GAME_SCORE_COLUMN,
                                PLAYER_POINT_DIFFERENTIAL_COLUMN,
                                PLAYER_HEIGHT_COLUMN,
                                PLAYER_FORCED_FUMBLES_COLUMN,
                                PLAYER_FUMBLES_RECOVERED_COLUMN,
                                PLAYER_FUMBLES_RECOVERED_YARDS_COLUMN,
                                PLAYER_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_OFFENSIVE_TWO_POINT_RETURNS_COLUMN,
                                PLAYER_OFFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLES_TOUCHDOWNS_COLUMN,
                                PLAYER_AVERAGE_GAIN_COLUMN,
                                PLAYER_COMPLETION_PERCENTAGE_COLUMN,
                                PLAYER_COMPLETIONS_COLUMN,
                                PLAYER_ESPN_QUARTERBACK_RATING_COLUMN,
                                PLAYER_INTERCEPTION_PERCENTAGE_COLUMN,
                                PLAYER_INTERCEPTIONS_COLUMN,
                                PLAYER_LONG_PASSING_COLUMN,
                                PLAYER_MISC_YARDS_COLUMN,
                                PLAYER_NET_PASSING_YARDS_COLUMN,
                                PLAYER_NET_TOTAL_YARDS_COLUMN,
                                PLAYER_PASSING_ATTEMPTS_COLUMN,
                                PLAYER_PASSING_BIG_PLAYS_COLUMN,
                                PLAYER_PASSING_FIRST_DOWNS_COLUMN,
                                PLAYER_PASSING_FUMBLES_COLUMN,
                                PLAYER_PASSING_FUMBLES_LOST_COLUMN,
                                PLAYER_PASSING_TOUCHDOWN_PERCENTAGE_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_COLUMN,
                                PLAYER_PASSING_YARDS_COLUMN,
                                PLAYER_PASSING_YARDS_AFTER_CATCH_COLUMN,
                                PLAYER_PASSING_YARDS_AT_CATCH_COLUMN,
                                PLAYER_QUARTERBACK_RATING_COLUMN,
                                PLAYER_SACKS_COLUMN,
                                PLAYER_SACKS_YARDS_LOST_COLUMN,
                                PLAYER_NET_PASSING_ATTEMPTS_COLUMN,
                                PLAYER_TOTAL_OFFENSIVE_PLAYS_COLUMN,
                                PLAYER_TOTAL_POINTS_COLUMN,
                                PLAYER_TOTAL_TOUCHDOWNS_COLUMN,
                                PLAYER_TOTAL_YARDS_COLUMN,
                                PLAYER_TOTAL_YARDS_FROM_SCRIMMAGE_COLUMN,
                                PLAYER_TWO_POINT_PASS_COLUMN,
                                PLAYER_TWO_POINT_PASS_ATTEMPT_COLUMN,
                                PLAYER_YARDS_PER_COMPLETION_COLUMN,
                                PLAYER_YARDS_PER_PASS_ATTEMPT_COLUMN,
                                PLAYER_NET_YARDS_PER_PASS_ATTEMPT_COLUMN,
                                PLAYER_ESPN_RUNNINGBACK_RATING_COLUMN,
                                PLAYER_LONG_RUSHING_COLUMN,
                                PLAYER_RUSHING_ATTEMPTS_COLUMN,
                                PLAYER_RUSHING_BIG_PLAYS_COLUMN,
                                PLAYER_RUSHING_FIRST_DOWNS_COLUMN,
                                PLAYER_RUSHING_FUMBLES_COLUMN,
                                PLAYER_RUSHING_FUMBLES_LOST_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_COLUMN,
                                PLAYER_RUSHING_YARDS_COLUMN,
                                PLAYER_STUFFS_COLUMN,
                                PLAYER_STUFF_YARDS_LOST,
                                PLAYER_TWO_POINT_RUSH_COLUMN,
                                PLAYER_TWO_POINT_RUSH_ATTEMPTS_COLUMN,
                                PLAYER_YARDS_PER_RUSH_ATTEMPT_COLUMN,
                                PLAYER_ESPN_WIDERECEIVER_COLUMN,
                                PLAYER_LONG_RECEPTION_COLUMN,
                                PLAYER_RECEIVING_BIG_PLAYS_COLUMN,
                                PLAYER_RECEIVING_FIRST_DOWNS_COLUMN,
                                PLAYER_RECEIVING_FUMBLES_COLUMN,
                                PLAYER_RECEIVING_FUMBLES_LOST_COLUMN,
                                PLAYER_RECEIVING_TARGETS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_COLUMN,
                                PLAYER_RECEIVING_YARDS_COLUMN,
                                PLAYER_RECEIVING_YARDS_AFTER_CATCH_COLUMN,
                                PLAYER_RECEIVING_YARDS_AT_CATCH_COLUMN,
                                PLAYER_RECEPTIONS_COLUMN,
                                PLAYER_TWO_POINT_RECEPTIONS_COLUMN,
                                PLAYER_TWO_POINT_RECEPTION_ATTEMPTS_COLUMN,
                                PLAYER_YARDS_PER_RECEPTION_COLUMN,
                                PLAYER_ASSIST_TACKLES_COLUMN,
                                PLAYER_AVERAGE_INTERCEPTION_YARDS_COLUMN,
                                PLAYER_AVERAGE_SACK_YARDS_COLUMN,
                                PLAYER_AVERAGE_STUFF_YARDS_COLUMN,
                                PLAYER_BLOCKED_FIELD_GOAL_TOUCHDOWNS_COLUMN,
                                PLAYER_BLOCKED_PUNT_TOUCHDOWNS_COLUMN,
                                PLAYER_DEFENSIVE_TOUCHDOWNS_COLUMN,
                                PLAYER_HURRIES_COLUMN,
                                PLAYER_KICKS_BLOCKED_COLUMN,
                                PLAYER_LONG_INTERCEPTION_COLUMN,
                                PLAYER_MISC_TOUCHDOWNS_COLUMN,
                                PLAYER_PASSES_BATTED_DOWN_COLUMN,
                                PLAYER_PASSES_DEFENDED_COLUMN,
                                PLAYER_QUARTERBACK_HITS_COLUMN,
                                PLAYER_SACKS_ASSISTED_COLUMN,
                                PLAYER_SACKS_UNASSISTED_COLUMN,
                                PLAYER_SACKS_YARDS_COLUMN,
                                PLAYER_SAFETIES_COLUMN,
                                PLAYER_SOLO_TACKLES_COLUMN,
                                PLAYER_STUFF_YARDS_COLUMN,
                                PLAYER_TACKLES_FOR_LOSS_COLUMN,
                                PLAYER_TACKLES_YARDS_LOST_COLUMN,
                                PLAYER_YARDS_ALLOWED_COLUMN,
                                PLAYER_POINTS_ALLOWED_COLUMN,
                                PLAYER_ONE_POINT_SAFETIES_MADE_COLUMN,
                                PLAYER_MISSED_FIELD_GOAL_RETURN_TD_COLUMN,
                                PLAYER_BLOCKED_PUNT_EZ_REC_TD_COLUMN,
                                PLAYER_INTERCEPTION_TOUCHDOWNS_COLUMN,
                                PLAYER_INTERCEPTION_YARDS_COLUMN,
                                PLAYER_AVERAGE_KICKOFF_RETURN_YARDS_COLUMN,
                                PLAYER_AVERAGE_KICKOFF_YARDS_COLUMN,
                                PLAYER_EXTRA_POINT_ATTEMPTS_COLUMN,
                                PLAYER_EXTRA_POINT_PERCENTAGE_COLUMN,
                                PLAYER_EXTRA_POINT_BLOCKED_COLUMN,
                                PLAYER_EXTRA_POINTS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_EXTRA_POINTS_MADE_COLUMN,
                                PLAYER_FAIR_CATCHES_COLUMN,
                                PLAYER_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_19_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_29_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_39_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_49_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_59_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_MAX_99_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPTS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_FIELD_GOAL_ATTEMPT_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_BLOCKED_COLUMN,
                                PLAYER_FIELD_GOALS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_19_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_29_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_39_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_49_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_59_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_MAX_99_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_ABOVE_50_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MADE_YARDS_COLUMN,
                                PLAYER_FIELD_GOALS_MISSED_YARDS_COLUMN,
                                PLAYER_KICKOFF_OUT_OF_BOUNDS_COLUMN,
                                PLAYER_KICKOFF_RETURNS_COLUMN,
                                PLAYER_KICKOFF_RETURNS_TOUCHDOWNS_COLUMN,
                                PLAYER_KICKOFF_RETURN_YARDS_COLUMN,
                                PLAYER_KICKOFFS_COLUMN,
                                PLAYER_KICKOFF_YARDS_COLUMN,
                                PLAYER_LONG_FIELD_GOAL_ATTEMPT_COLUMN,
                                PLAYER_LONG_FIELD_GOAL_MADE_COLUMN,
                                PLAYER_LONG_KICKOFF_COLUMN,
                                PLAYER_TOTAL_KICKING_POINTS_COLUMN,
                                PLAYER_TOUCHBACK_PERCENTAGE_COLUMN,
                                PLAYER_TOUCHBACKS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLE_RETURNS_COLUMN,
                                PLAYER_DEFENSIVE_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_FUMBLE_RECOVERIES_COLUMN,
                                PLAYER_FUMBLE_RECOVERY_YARDS_COLUMN,
                                PLAYER_KICK_RETURN_FAIR_CATCHES_COLUMN,
                                PLAYER_KICK_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_KICK_RETURN_FUMBLES_COLUMN,
                                PLAYER_KICK_RETURN_FUMBLES_LOST_COLUMN,
                                PLAYER_KICK_RETURNS_COLUMN,
                                PLAYER_KICK_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_KICK_RETURN_YARDS_COLUMN,
                                PLAYER_LONG_KICK_RETURN_COLUMN,
                                PLAYER_LONG_PUNT_RETURN_COLUMN,
                                PLAYER_MISC_FUMBLE_RETURNS_COLUMN,
                                PLAYER_MISC_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_OPPOSITION_FUMBLE_RECOVERIES_COLUMN,
                                PLAYER_OPPOSITION_FUMBLE_RECOVERY_YARDS_COLUMN,
                                PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN,
                                PLAYER_OPPOSITION_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_PUNT_RETURN_FAIR_CATCHES_COLUMN,
                                PLAYER_PUNT_RETURN_FAIR_CATCH_PERCENTAGE_COLUMN,
                                PLAYER_PUNT_RETURN_FUMBLES_COLUMN,
                                PLAYER_PUNT_RETURN_FUMBLES_LOST_COLUMN,
                                PLAYER_PUNT_RETURNS_COLUMN,
                                PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_10_COLUMN,
                                PLAYER_PUNT_RETURNS_STARTED_INSIDE_THE_20_COLUMN,
                                PLAYER_PUNT_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_PUNT_RETURN_YARDS_COLUMN,
                                PLAYER_SPECIAL_TEAM_FUMBLE_RETURNS_COLUMN,
                                PLAYER_SPECIAL_TEAM_FUMBLE_RETURN_YARDS_COLUMN,
                                PLAYER_YARDS_PER_KICK_RETURN_COLUMN,
                                PLAYER_YARDS_PER_RETURN_COLUMN,
                                PLAYER_AVERAGE_PUNT_RETURN_YARDS_COLUMN,
                                PLAYER_GROSS_AVERAGE_PUNT_YARDS_COLUMN,
                                PLAYER_LONG_PUNT_COLUMN,
                                PLAYER_NET_AVERAGE_PUNT_YARDS_COLUMN,
                                PLAYER_PUNTS_COLUMN,
                                PLAYER_PUNTS_BLOCKED_COLUMN,
                                PLAYER_PUNTS_BLOCKED_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_INSIDE_10_COLUMN,
                                PLAYER_PUNTS_INSIDE_10_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_INSIDE_20_COLUMN,
                                PLAYER_PUNTS_INSIDE_20_PERCENTAGE_COLUMN,
                                PLAYER_PUNTS_OVER_50_COLUMN,
                                PLAYER_PUNT_YARDS_COLUMN,
                                PLAYER_DEFENSIVE_POINTS_COLUMN,
                                PLAYER_MISC_POINTS_COLUMN,
                                PLAYER_RETURN_TOUCHDOWNS_COLUMN,
                                PLAYER_TOTAL_TWO_POINT_CONVERSIONS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_PASSING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_RECEIVING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_9_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_19_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_29_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_39_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_49_YARDS_COLUMN,
                                PLAYER_RUSHING_TOUCHDOWNS_ABOVE_50_YARDS_COLUMN,
                                PLAYER_PENALTIES_IN_MINUTES_COLUMN,
                                PLAYER_EVEN_STRENGTH_GOALS_COLUMN,
                                PLAYER_POWER_PLAY_GOALS_COLUMN,
                                PLAYER_SHORT_HANDED_GOALS_COLUMN,
                                PLAYER_GAME_WINNING_GOALS_COLUMN,
                                PLAYER_EVEN_STRENGTH_ASSISTS_COLUMN,
                                PLAYER_POWER_PLAY_ASSISTS_COLUMN,
                                PLAYER_SHORT_HANDED_ASSISTS_COLUMN,
                                PLAYER_SHOTS_ON_GOAL_COLUMN,
                                PLAYER_SHOOTING_PERCENTAGE_COLUMN,
                                PLAYER_SHIFTS_COLUMN,
                                PLAYER_TIME_ON_ICE_COLUMN,
                                PLAYER_DECISION_COLUMN,
                                PLAYER_GOALS_AGAINST_COLUMN,
                                PLAYER_SHOTS_AGAINST_COLUMN,
                                PLAYER_SAVES_COLUMN,
                                PLAYER_SAVE_PERCENTAGE_COLUMN,
                                PLAYER_SHUTOUTS_COLUMN,
                                PLAYER_INDIVIDUAL_CORSI_FOR_EVENTS_COLUMN,
                                PLAYER_ON_SHOT_ICE_FOR_EVENTS_COLUMN,
                                PLAYER_ON_SHOT_ICE_AGAINST_EVENTS_COLUMN,
                                PLAYER_CORSI_FOR_PERCENTAGE_COLUMN,
                                PLAYER_RELATIVE_CORSI_FOR_PERCENTAGE_COLUMN,
                                PLAYER_OFFENSIVE_ZONE_STARTS_COLUMN,
                                PLAYER_DEFENSIVE_ZONE_STARTS_COLUMN,
                                PLAYER_OFFENSIVE_ZONE_START_PERCENTAGE_COLUMN,
                                PLAYER_HITS_COLUMN,
                                PLAYER_TRUE_SHOOTING_PERCENTAGE_COLUMN,
                                PLAYER_AT_BATS_COLUMN,
                                PLAYER_RUNS_SCORED_COLUMN,
                                PLAYER_RUNS_BATTED_IN_COLUMN,
                                PLAYER_BASES_ON_BALLS_COLUMN,
                                PLAYER_STRIKEOUTS_COLUMN,
                                PLAYER_PLATE_APPEARANCES_COLUMN,
                                PLAYER_HITS_AT_BATS_COLUMN,
                                PLAYER_OBP_COLUMN,
                                PLAYER_SLG_COLUMN,
                                PLAYER_OPS_COLUMN,
                                PLAYER_PITCHES_COLUMN,
                                PLAYER_STRIKES_COLUMN,
                                PLAYER_WIN_PROBABILITY_ADDED_COLUMN,
                                PLAYER_AVERAGE_LEVERAGE_INDEX_COLUMN,
                                PLAYER_WPA_PLUS_COLUMN,
                                PLAYER_WPA_MINUS_COLUMN,
                                PLAYER_CWPA_COLUMN,
                                PLAYER_ACLI_COLUMN,
                                PLAYER_RE24_COLUMN,
                                PLAYER_PUTOUTS_COLUMN,
                                PLAYER_INNINGS_PITCHED_COLUMN,
                                PLAYER_EARNED_RUNS_COLUMN,
                                PLAYER_HOME_RUNS_COLUMN,
                                PLAYER_ERA_COLUMN,
                                PLAYER_BATTERS_FACED_COLUMN,
                                PLAYER_STRIKES_BY_CONTACT_COLUMN,
                                PLAYER_STRIKES_SWINGING_COLUMN,
                                PLAYER_STRIKES_LOOKING_COLUMN,
                                PLAYER_GROUND_BALLS_COLUMN,
                                PLAYER_FLY_BALLS_COLUMN,
                                PLAYER_LINE_DRIVES_COLUMN,
                                PLAYER_INHERITED_RUNNERS_COLUMN,
                                PLAYER_INHERITED_SCORES_COLUMN,
                                PLAYER_EFFECTIVE_FIELD_GOAL_PERCENTAGE_COLUMN,
                                PLAYER_PENALTY_KICKS_MADE_COLUMN,
                                PLAYER_PENALTY_KICKS_ATTEMPTED_COLUMN,
                                PLAYER_SHOTS_TOTAL_COLUMN,
                                PLAYER_SHOTS_ON_TARGET_COLUMN,
                                PLAYER_YELLOW_CARDS_COLUMN,
                                PLAYER_RED_CARDS_COLUMN,
                                PLAYER_TOUCHES_COLUMN,
                                PLAYER_EXPECTED_GOALS_COLUMN,
                                PLAYER_NON_PENALTY_EXPECTED_GOALS_COLUMN,
                                PLAYER_EXPECTED_ASSISTED_GOALS_COLUMN,
                                PLAYER_SHOT_CREATING_ACTIONS_COLUMN,
                                PLAYER_GOAL_CREATING_ACTIONS_COLUMN,
                                PLAYER_PASSES_COMPLETED_COLUMN,
                                PLAYER_PASSES_ATTEMPTED_COLUMN,
                                PLAYER_PASS_COMPLETION_COLUMN,
                                PLAYER_PROGRESSIVE_PASSES_COLUMN,
                                PLAYER_CARRIES_COLUMN,
                                PLAYER_PROGRESSIVE_CARRIES_COLUMN,
                                PLAYER_TAKE_ONS_ATTEMPTED_COLUMN,
                                PLAYER_SUCCESSFUL_TAKE_ONS_COLUMN,
                                PLAYER_TOTAL_PASSING_DISTANCE_COLUMN,
                                PLAYER_PROGRESSIVE_PASSING_DISTANCE_COLUMN,
                                PLAYER_PASSES_COMPLETED_SHORT_COLUMN,
                                PLAYER_PASSES_ATTEMPTED_SHORT_COLUMN,
                                PLAYER_PASS_COMPLETION_SHORT_COLUMN,
                                PLAYER_PASSES_COMPLETED_MEDIUM_COLUMN,
                                PLAYER_PASSES_ATTEMPTED_MEDIUM_COLUMN,
                                PLAYER_PASS_COMPLETION_MEDIUM_COLUMN,
                                PLAYER_PASSES_COMPLETED_LONG_COLUMN,
                                PLAYER_PASSES_ATTEMPTED_LONG_COLUMN,
                                PLAYER_PASS_COMPLETION_LONG_COLUMN,
                                PLAYER_EXPECTED_ASSISTS_COLUMN,
                                PLAYER_KEY_PASSES_COLUMN,
                                PLAYER_PASSES_INTO_FINAL_THIRD_COLUMN,
                                PLAYER_PASSES_INTO_PENALTY_AREA_COLUMN,
                                PLAYER_CROSSES_INTO_PENALTY_AREA_COLUMN,
                                PLAYER_LIVE_BALL_PASSES_COLUMN,
                                PLAYER_DEAD_BALL_PASSES_COLUMN,
                                PLAYER_PASSES_FROM_FREE_KICKS_COLUMN,
                                PLAYER_THROUGH_BALLS_COLUMN,
                                PLAYER_SWITCHES_COLUNM,
                                PLAYER_CROSSES_COLUMN,
                                PLAYER_THROW_INS_TAKEN_COLUMN,
                                PLAYER_CORNER_KICKS_COLUMN,
                                PLAYER_INSWINGING_CORNER_KICKS_COLUMN,
                                PLAYER_OUTSWINGING_CORNER_KICKS_COLUMN,
                                PLAYER_STRAIGHT_CORNER_KICKS_COLUMN,
                                PLAYER_PASSES_OFFSIDE_COLUMN,
                                PLAYER_PASSES_BLOCKED_COLUMN,
                                PLAYER_TACKLES_WON_COLUMN,
                                PLAYER_TACKLES_IN_DEFENSIVE_THIRD_COLUMN,
                                PLAYER_TACKLES_IN_MIDDLE_THIRD_COLUMN,
                                PLAYER_TACKLES_IN_ATTACKING_THIRD_COLUMN,
                                PLAYER_DRIBBLERS_TACKLED_COLUMN,
                                PLAYER_DRIBBLES_CHALLENGED_COLUMN,
                                PLAYER_PERCENT_OF_DRIBBLERS_TACKLED_COLUMN,
                                PLAYER_CHALLENGES_LOST_COLUMN,
                                PLAYER_SHOTS_BLOCKED_COLUMN,
                                PLAYER_TACKLES_PLUS_INTERCEPTIONS_COLUMN,
                                PLAYER_ERRORS_COLUMN,
                                PLAYER_TOUCHES_IN_DEFENSIVE_PENALTY_AREA_COLUMN,
                                PLAYER_TOUCHES_IN_DEFENSIVE_THIRD_COLUMN,
                                PLAYER_TOUCHES_IN_MIDDLE_THIRD_COLUMN,
                                PLAYER_TOUCHES_IN_ATTACKING_THIRD_COLUMN,
                                PLAYER_TOUCHES_IN_ATTACKING_PENALTY_AREA_COLUMN,
                                PLAYER_LIVE_BALL_TOUCHES_COLUMN,
                                PLAYER_SUCCESSFUL_TAKE_ON_PERCENTAGE_COLUMN,
                                PLAYER_TIMES_TACKLED_DURING_TAKE_ONS_COLUMN,
                                PLAYER_TACKLED_DURING_TAKE_ON_PERCENTAGE_COLUMN,
                                PLAYER_TOTAL_CARRYING_DISTANCE_COLUMN,
                                PLAYER_PROGRESSIVE_CARRYING_DISTANCE_COLUMN,
                                PLAYER_CARRIES_INTO_FINAL_THIRD_COLUMN,
                                PLAYER_CARRIES_INTO_PENALTY_AREA_COLUMN,
                                PLAYER_MISCONTROLS_COLUMN,
                                PLAYER_DISPOSSESSED_COLUMN,
                                PLAYER_PASSES_RECEIVED_COLUMN,
                                PLAYER_PROGRESSIVE_PASSES_RECEIVED_COLUMN,
                                PLAYER_SECOND_YELLOW_CARD_COLUMN,
                                PLAYER_FOULS_COMMITTED_COLUMN,
                                PLAYER_FOULS_DRAWN_COLUMN,
                                PLAYER_OFFSIDES_COLUMN,
                                PLAYER_PENALTY_KICKS_WON_COLUMN,
                                PLAYER_PENALTY_KICKS_CONCEDED_COLUMN,
                                PLAYER_OWN_GOALS_COLUMN,
                                PLAYER_BALL_RECOVERIES_COLUMN,
                                PLAYER_AERIALS_WON_COLUMN,
                                PLAYER_AERIALS_LOST_COLUMN,
                                PLAYER_PERCENTAGE_OF_AERIALS_WON_COLUMN,
                                PLAYER_SHOTS_ON_TARGET_AGAINST_COLUMN,
                                PLAYER_POST_SHOT_EXPECTED_GOALS_COLUMN,
                                PLAYER_PASSES_ATTEMPTED_MINUS_GOAL_KICKS_COLUMN,
                                PLAYER_THROWS_ATTEMPTED_COLUMN,
                                PLAYER_PERCENTAGE_OF_PASSES_THAT_WERE_LAUNCHED_COLUMN,
                                PLAYER_AVERAGE_PASS_LENGTH_COLUMN,
                                PLAYER_GOAL_KICKS_ATTEMPTED_COLUMN,
                                PLAYER_PERCENTAGE_OF_GOAL_KICKS_THAT_WERE_LAUNCHED_COLUMN,
                                PLAYER_AVERAGE_GOAL_KICK_LENGTH_COLUMN,
                                PLAYER_CROSSES_FACED_COLUMN,
                                PLAYER_CROSSES_STOPPED_COLUMN,
                                PLAYER_PERCENTAGE_CROSSES_STOPPED_COLUMN,
                                PLAYER_DEFENSIVE_ACTIONS_OUTSIDE_PENALTY_AREA_COLUMN,
                                PLAYER_AVERAGE_DISTANCE_OF_DEFENSIVE_ACTIONS_COLUMN,
                                PLAYER_THREE_POINT_ATTEMPT_RATE_COLUMN,
                                PLAYER_BATTING_STYLE_COLUMN,
                                PLAYER_BOWLING_STYLE_COLUMN,
                                PLAYER_PLAYING_ROLES_COLUMN,
                                PLAYER_RUNS_COLUMN,
                                PLAYER_BALLS_COLUMN,
                                PLAYER_FOURS_COLUMN,
                                PLAYER_SIXES_COLUMN,
                                PLAYER_STRIKERATE_COLUMN,
                                PLAYER_FALL_OF_WICKET_ORDER_COLUMN,
                                PLAYER_FALL_OF_WICKET_NUM_COLUMN,
                                PLAYER_FALL_OF_WICKET_RUNS_COLUMN,
                                PLAYER_FALL_OF_WICKET_BALLS_COLUMN,
                                PLAYER_FALL_OF_WICKET_OVERS_COLUMN,
                                PLAYER_FALL_OF_WICKET_OVER_NUMBER_COLUMN,
                                PLAYER_BALL_OVER_ACTUAL_COLUMN,
                                PLAYER_BALL_OVER_UNIQUE_COLUMN,
                                PLAYER_BALL_TOTAL_RUNS_COLUMN,
                                PLAYER_BALL_BATSMAN_RUNS_COLUMN,
                                PLAYER_OVERS_COLUMN,
                                PLAYER_MAIDENS_COLUMN,
                                PLAYER_CONCEDED_COLUMN,
                                PLAYER_WICKETS_COLUMN,
                                PLAYER_ECONOMY_COLUMN,
                                PLAYER_RUNS_PER_BALL_COLUMN,
                                PLAYER_DOTS_COLUMN,
                                PLAYER_WIDES_COLUMN,
                                PLAYER_NO_BALLS_COLUMN,
                                PLAYER_FREE_THROW_ATTEMPT_RATE_COLUMN,
                                PLAYER_OFFENSIVE_REBOUND_PERCENTAGE_COLUMN,
                                PLAYER_DEFENSIVE_REBOUND_PERCENTAGE_COLUMN,
                                PLAYER_TOTAL_REBOUND_PERCENTAGE_COLUMN,
                                PLAYER_ASSIST_PERCENTAGE_COLUMN,
                                PLAYER_STEAL_PERCENTAGE_COLUMN,
                                PLAYER_BLOCK_PERCENTAGE_COLUMN,
                                PLAYER_TURNOVER_PERCENTAGE_COLUMN,
                                PLAYER_USAGE_PERCENTAGE_COLUMN,
                                PLAYER_OFFENSIVE_RATING_COLUMN,
                                PLAYER_DEFENSIVE_RATING_COLUMN,
                                PLAYER_BOX_PLUS_MINUS_COLUMN,
                            ]
                        ],
                        player_column_prefix(i, x),
                        points_column=team_points_column(i),
                        field_goals_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_FIELD_GOALS_COLUMN]
                        ),
                        assists_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_ASSISTS_COLUMN]
                        ),
                        field_goals_attempted_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_FIELD_GOALS_ATTEMPTED_COLUMN,
                            ]
                        ),
                        offensive_rebounds_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_OFFENSIVE_REBOUNDS_COLUMN,
                            ]
                        ),
                        turnovers_column=DELIMITER.join(
                            [player_column_prefix(i, x), PLAYER_TURNOVERS_COLUMN]
                        ),
                        team_identifier_column=team_identifier_column(i),
                        birth_date_column=DELIMITER.join(
                            [
                                player_column_prefix(i, x),
                                PLAYER_BIRTH_DATE_COLUMN,
                            ]
                        ),
                        image_columns=[PLAYER_HEADSHOT_COLUMN],
                    )
                    for x in range(player_count)
                ]
            )
            for player_id in range(player_count):
                datetime_columns.add(
                    DELIMITER.join(
                        [player_column_prefix(i, player_id), PLAYER_BIRTH_DATE_COLUMN]
                    )
                )
            coach_count = find_coach_count(df, i)
            identifiers.extend(
                [
                    Identifier(
                        entity_type=EntityType.COACH,
                        column=coach_identifier_column(i, x),
                        feature_columns=[],
                        column_prefix=coach_column_prefix(i, x),
                        points_column=team_points_column(i),
                        team_identifier_column=team_identifier_column(i),
                    )
                    for x in range(coach_count)
                ]
            )
        df_processed = process(
            df,
            GAME_DT_COLUMN,
            identifiers,
            [None]
            + [datetime.timedelta(days=365 * i) for i in [1, 2, 4, 8]]
            + [datetime.timedelta(days=i * 7) for i in [2, 4]],
            df.attrs[str(FieldType.CATEGORICAL)],
            use_bets_features=False,
            use_news_features=True,
            datetime_columns=datetime_columns,
            use_players_feature=True,
        )
        df_processed.to_parquet(df_cache_path)
        return df_processed

    def _calculate_embedding_columns(self, df: pd.DataFrame) -> list[list[str]]:
        team_count = find_team_count(df)

        embedding_cols = []
        for i in range(team_count):
            col_prefix = team_column_prefix(i)
            embedding_cols.append(
                [
                    x
                    for x in df.columns.values.tolist()
                    if x.startswith(col_prefix) and is_embedding_column(x)
                ]
            )

        return embedding_cols
