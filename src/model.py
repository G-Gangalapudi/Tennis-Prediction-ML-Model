"""Train and run an XGBoost classifier on pre-match ATP features."""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


NUMERIC_FEATURES = [
    "ranking_diff",
    "player1_rank",
    "player2_rank",
    "player1_recent_win_rate_5",
    "player2_recent_win_rate_5",
    "player1_recent_win_rate_10",
    "player2_recent_win_rate_10",
    "player1_surface_win_rate",
    "player2_surface_win_rate",
    "head_to_head",
    "age_diff",
    "height_diff",
]

CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]


def prepare_features(df):
    """Select pre-match inputs, treating absent feature columns as missing."""
    features = df.reindex(columns=NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("string").fillna("__missing__")
    return features


def train_model(X_train, y_train):
    """Fit the encoder and classifier using the caller's training rows."""
    preprocessor = ColumnTransformer(
        [
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ],
    )
    classifier = XGBClassifier(
        objective="binary:logistic",
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=2,
    )
    model = Pipeline([("preprocess", preprocessor), ("classifier", classifier)])
    return model.fit(prepare_features(X_train), y_train)


def predict_model(model, X_test):
    """Return 1 for a predicted Player 1 win or 0 for a Player 2 win."""
    return model.predict(prepare_features(X_test))


def predict_probabilities(model, X_test):
    """Return the probability that Player 1 (the lower ATP ID) wins."""
    return model.predict_proba(prepare_features(X_test))[:, 1]
