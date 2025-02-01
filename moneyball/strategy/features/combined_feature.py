"""A feature extractor combining many sub feature extractors."""

import numpy as np
import pandas as pd
from sportsball.data.field_type import FieldType  # type: ignore
from sportsball.data.game_model import END_DT_COLUMN  # type: ignore
from sportsball.data.game_model import GAME_DT_COLUMN

from .datetime_feature import DatetimeFeature
from .datetimesub_feature import DatetimeSubFeature
from .feature import Feature
from .lag_feature import LagFeature
from .min_feature import MinFeature
from .offensive_efficiency_feature import OffensiveEfficiencyFeature
from .ordinal_feature import OrdinalFeature
from .skill_feature import SkillFeature
from .sma_feature import SMAFeature
from .total_feature import TotalFeature


class CombinedFeature(Feature):
    """Combined feature extractor class."""

    # pylint: disable=too-few-public-methods

    def __init__(
        self,
        pretrain_features: list[Feature] | None = None,
        posttrain_features: list[Feature] | None = None,
    ) -> None:
        super().__init__()
        if pretrain_features is None:
            pretrain_features = [
                SkillFeature(year_slices=[None, 1, 2, 4, 8]),
                LagFeature(),
                TotalFeature(),
                MinFeature(),
                SMAFeature(),
                DatetimeSubFeature(END_DT_COLUMN, GAME_DT_COLUMN, "m"),
                OffensiveEfficiencyFeature(),
            ]
        if posttrain_features is None:
            posttrain_features = [
                DatetimeFeature(),
                OrdinalFeature(),
            ]
        self._pretrain_features = pretrain_features
        self._posttrain_features = posttrain_features

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        for feature in self._pretrain_features:
            df = feature.process(df)
        df = df[list(set(df.columns.values) - set(df.attrs[str(FieldType.LOOKAHEAD)]))]
        for feature in self._posttrain_features:
            df = feature.process(df)
        return df.replace([np.inf, -np.inf], np.nan)
