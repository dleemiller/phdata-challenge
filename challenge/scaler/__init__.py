from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer

from .sinusoidal_scaler import SinusoidalTransformer
from .scaler_999 import Scaler999


class TransformerConfigBase:

    @classmethod
    def export(cls):
        assert hasattr(cls, "transformer")
        assert hasattr(cls, "features")
        return (cls.__name__, cls.transformer, cls.features)


class OneHotConfig(TransformerConfigBase):

    @classmethod
    def new(cls):
        cls.transformer = OneHotEncoder(handle_unknown="ignore")
        cls.features = ["b1", "b2"]
        return cls.export()


class SinusoidalMonthConfig(TransformerConfigBase):
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
        cls.features = ["month"]
        return cls.export()


class SinusoidalDayConfig(TransformerConfigBase):
    column_map = {"mon": 1, "tue": 2, "wed": 3, "thu": 4, "fri": 5}

    @classmethod
    def new(cls):
        cls.transformer = SinusoidalTransformer(5, cls.column_map)
        cls.features = ["dow"]
        return cls.export()


class Scaler999Config(TransformerConfigBase):
    """Maps 999 to 1 and everything else to 0!"""

    @classmethod
    def new(cls):
        cls.transformer = Scaler999()
        cls.features = ["n4"]
        return cls.export()


preprocessor = ColumnTransformer(
    transformers=[
        OneHotConfig.new(),
        SinusoidalMonthConfig.new(),
        SinusoidalDayConfig.new(),
        ("drop", "drop", ["c10"]),
    ]
)
# # Define the transformations for each column type
# numeric_features = ['age']
# numeric_transformer = Pipeline(steps=[
#     ('imputer', SimpleImputer(strategy='median')),
#     ('scaler', StandardScaler())])
#
# b_features = ['b1', 'b2']
# b_transformer = MinMaxScaler()
#
# categorical_features = ['employment', 'school', 'dow']
# categorical_transformer = OneHotEncoder(handle_unknown='ignore')
#
# c_features = ['c10', 'c3', 'c4', 'c8']
# c_transformer = SimpleImputer(strategy='mean')
#
# # Column transformer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', numeric_transformer, numeric_features),
#         ('b', b_transformer, b_features),
#         ('cat', categorical_transformer, categorical_features),
#         ('c', c_transformer, c_features),
#         ('drop', 'drop', ['i1', 'i2', 'successful_sell'])
#     ])
#
# # Create the full pipeline
# pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
