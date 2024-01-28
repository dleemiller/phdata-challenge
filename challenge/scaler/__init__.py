from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
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
    features = ["c3", "c4", "c8", "marriage-status", "school", "employment"]
    transformer = OneHotEncoder(handle_unknown="ignore")


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


class ExtractBandConfig(TransformerConfigBase):
    """Extract band from between two values as feature"""

    features = ["i3"]
    transformer = ExtractBand(-40.8, -37.5)


class KBinsUniformConfig(TransformerConfigBase):
    features = ["i1"]
    transformer = KBinsDiscretizer(
        n_bins=10, encode="onehot", strategy="uniform", subsample=None
    )


class KBinsKMeansConfig(TransformerConfigBase):
    features = ["i2"]
    transformer = KBinsDiscretizer(
        n_bins=10, encode="onehot", strategy="kmeans", subsample=None
    )


class PCAConfig(TransformerConfigBase):
    features = ["i1", "i2", "i4", "i5"]
    transformer = PCA(n_components=1)


preprocessor = ColumnTransformer(
    transformers=[
        OneHotConfig.export(),
        SinusoidalMonthConfig.new(),
        SinusoidalDayConfig.new(),
        Scaler999Config.export(),
        ExtractBandConfig.export(),
        KBinsUniformConfig.export(),
        # KBinsKMeansConfig.export(),
        PCAConfig.export(),
        ("pass", "passthrough", ["n2", "n6", "age"]),
        ("drop", "drop", ["c10", "b1", "b2", "n3", "n5", "successful_sell"]),
    ]
)
