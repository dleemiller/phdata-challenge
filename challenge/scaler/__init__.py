from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    KBinsDiscretizer,
    OrdinalEncoder,
    QuantileTransformer,
)
from sklearn.decomposition import PCA

from .sinusoidal_scaler import SinusoidalTransformer
from .scaler_999 import Scaler999
from .band_extraction import ExtractBand


class TransformerConfigBase:
    """A base class for exporting a configuration"""

    @classmethod
    def export(cls):
        assert hasattr(cls, "transformer")
        assert hasattr(cls, "features")
        return (cls.__name__, cls.transformer, cls.features)


class OneHotConfig(TransformerConfigBase):
    features = ["marriage-status", "employment", "school", "c8", "c3", "month", "dow"]
    transformer = OneHotEncoder(
        handle_unknown="infrequent_if_exist", min_frequency=0.05
    )


class OrdinalConfig(TransformerConfigBase):
    features = ["b1", "b2", "c4", "c8"]
    transformer = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-1,
        min_frequency=0.05,
    )


class SinusoidalMonthConfig(TransformerConfigBase):
    features = ["month"]
    column_map = {
        "jan": 1,
        "feb": 2,
        "mar": 3,
        "apr": 4,
        "may": 5,
        "jun": 6,
        "jul": 7,
        "aug": 8,
        "sep": 9,
        "oct": 10,
        "nov": 11,
        "dec": 12,
    }

    @classmethod
    def new(cls):
        cls.transformer = SinusoidalTransformer(12, cls.column_map)
        return cls.export()


class SinusoidalDayConfig(TransformerConfigBase):
    features = ["dow"]
    column_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}

    @classmethod
    def new(cls):
        cls.transformer = SinusoidalTransformer(5, cls.column_map)
        return cls.export()


class Scaler999Config(TransformerConfigBase):
    """Maps 999 to 1 and everything else to 0!"""

    features = ["n4"]
    transformer = Scaler999()


class ExtractI1BandConfig(TransformerConfigBase):
    """Extract band from between two values as feature"""

    features = ["i1"]
    transformer = ExtractBand(-1.8, -1.1)


class ExtractI3BandConfig(TransformerConfigBase):
    """Extract band from between two values as feature"""

    features = ["i3"]
    transformer = ExtractBand(-40.8, -37.5)


class KBinsConfig(TransformerConfigBase):
    features = ["i2"]
    transformer = KBinsDiscretizer(
        n_bins=10, encode="ordinal", strategy="uniform", subsample=None
    )


class PCAConfig(TransformerConfigBase):
    features = ["i1", "i2", "i4", "i5", "n6"]
    transformer = Pipeline(
        [
            ("scaler", QuantileTransformer(output_distribution="normal")),
            ("pca", PCA(n_components=2)),
        ]
    )


class QuantileConfig(TransformerConfigBase):
    features = ["n2", "n6", "age", "i1", "i2", "i3", "i4", "i5"]
    transformer = QuantileTransformer(output_distribution="normal")


preprocessor = ColumnTransformer(
    transformers=[
        OneHotConfig.export(),
        OrdinalConfig.export(),
        # SinusoidalMonthConfig.new(),
        # SinusoidalDayConfig.new(),
        Scaler999Config.export(),
        ExtractI1BandConfig.export(),
        ExtractI3BandConfig.export(),
        KBinsConfig.export(),
        # PCAConfig.export(),
        QuantileConfig.export(),
        ("pass", "passthrough", ["n4"]),
        ("drop", "drop", ["c10", "n3", "n5", "successful_sell"]),
    ]
)
