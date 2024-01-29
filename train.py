import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder

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

    grid_search = GridSearchCV(
        pipeline, param_grid, cv=cv, n_jobs=n_jobs, verbose=2, scoring="f1_macro"
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best F1 Score:", best_score)

    if not os.path.exists("artifacts"):
        os.mkdir("artifacts")

    print("Best parameters:", best_params)
    with open("artifacts/best_params.json", "w") as fh:
        json.dump(best_params, fh, indent=4)


def print_important_features(pipeline):
    feature_names = []
    for name, transformer, column in preprocessor.transformers_[:-1]:
        if transformer in ["drop"]:
            continue
        elif transformer in ["passthrough"]:
            feature_names.append(column)
            continue

        print(transformer.get_feature_names_out(input_features=column))
        feature_names.extend(transformer.get_feature_names_out(input_features=column))

    model = pipeline.named_steps["classifier"]
    importances = model.feature_importances_
    feature_importances = pd.DataFrame(
        importances, index=feature_names, columns=["Importance"]
    ).sort_values("Importance", ascending=False)

    print(feature_importances)


def save_eval_metrics(pipeline):
    # save the confusion matrix
    y_pred = pipeline.predict(X_test)
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)

    sns.heatmap(conf_matrix, annot=True, fmt="g", cmap="Blues")
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix")
    plt.savefig("artifacts/conf_matrix.png")

    report = classification_report(y_test, y_pred)
    with open("artifacts/classification_report.txt", "w") as file:
        file.write(report)


def train(args):
    if not os.path.exists("artifacts/best_params.json"):
        raise FileNotFoundError("Run gridsearch before training.")

    with open("artifacts/best_params.json", "r") as fh:
        params = json.load(fh)

    # set the parameters from gridsearch
    pipeline.set_params(**params)

    # fit the model
    pipeline.fit(X_train, y_train)
    # if args.print_important:
    #    print_important_features(pipeline)

    # save artifacts
    save_eval_metrics(pipeline)
    joblib.dump(pipeline, "artifacts/challenge_pipeline.pkl")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Model training and grid search script"
    )
    subparsers = parser.add_subparsers(help="sub-command help")
    parser_train = subparsers.add_parser("train", help="train help")
    parser_train.set_defaults(func=train)

    parser_search = subparsers.add_parser("search", help="gridsearch help")
    parser_search.add_argument(
        "--file", type=str, help="File path for grid parameters in JSON"
    )
    parser_search.add_argument(
        "--print_important",
        action="store_true",
        default=False,
        help="Print important features",
    )

    parser_search.set_defaults(func=search)

    args = parser.parse_args()
    args.func(args)

# sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues')
# plt.xlabel('Predicted labels')
# plt.ylabel('True labels')
# plt.title('Confusion Matrix')
# plt.show()
