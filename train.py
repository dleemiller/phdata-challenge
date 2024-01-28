import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

from challenge.scaler import preprocessor
from challenge.data import df

# get target data
y = df.successful_sell.factorize()[0]

# train/test split the data
X_train, X_test, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

# create the training pipeline
pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier())]
)


def search(args, cv=3, n_jobs=-1):
    with open(args.file, "r") as fh:
        import json

        param_grid = json.load(fh)

    grid_search = GridSearchCV(pipeline, param_grid, cv=cv, n_jobs=n_jobs, verbose=2)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")

    print("Best parameters:", best_params)
    with open("best_params.json", "w") as fh:
        json.dump(best_params, fh)


def train():
    # fit the model
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)

    joblib.dump(pipeline, "challenge_pipeline.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Model training and grid search script"
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.set_defaults(func=train)

    parser_search = subparsers.add_parser("gridsearch", help="gridsearch help")
    parser_search.add_argument(
        "--file", type=str, help="File path for grid parameters in JSON"
    )
    parser_search.set_defaults(func=search)

    args = parser.parse_args()
    args.func(args)

# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()
