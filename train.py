from challenge.scaler import preprocessor
from challenge.data import df


preprocessor.fit(df)
data = preprocessor.transform(df)
print(data)
print(data.shape)
