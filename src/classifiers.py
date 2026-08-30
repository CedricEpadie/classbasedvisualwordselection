"""Common `ClassifierWrapper` interface around all six classifiers, so the
pipeline trains/evaluates them identically and interchangeably.

All classifiers use scikit-learn/XGBoost default hyperparameters, per the
spec, with the single documented exception of `LogisticRegression.max_iter`
raised for convergence.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier  # noqa: F401 (kept for future extension)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  # pragma: no cover
    XGBClassifier = None


class ClassifierWrapper:
    """Uniform fit / predict / predict_proba interface, plus the class
    ordering needed to align `predict_proba` columns with metric
    computation later on.

    `needs_label_encoding`: some estimators (XGBoost's sklearn API in
    particular) require integer labels in [0, n_classes) and reject
    string/categorical `y` outright. When set, this wrapper transparently
    label-encodes on `fit` and decodes back to the original string labels
    on `predict`, so callers never need to know the estimator's quirks.
    """

    def __init__(self, name: str, estimator: Any, needs_label_encoding: bool = False):
        self.name = name
        self.estimator = estimator
        self.needs_label_encoding = needs_label_encoding
        self.classes_: Optional[np.ndarray] = None
        self._label_encoder = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassifierWrapper":
        if self.needs_label_encoding:
            from sklearn.preprocessing import LabelEncoder

            self._label_encoder = LabelEncoder()
            y_encoded = self._label_encoder.fit_transform(y)
            self.estimator.fit(X, y_encoded)
            # Keep classes_ in the original (string) label space so
            # downstream code (predictions dataframe, metrics) never sees
            # the internal integer encoding.
            self.classes_ = self._label_encoder.classes_
        else:
            self.estimator.fit(X, y)
            self.classes_ = getattr(self.estimator, "classes_", np.unique(y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        y_pred = self.estimator.predict(X)
        if self.needs_label_encoding:
            return self._label_encoder.inverse_transform(y_pred)
        return y_pred

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.estimator, "predict_proba"):
            # Columns already line up with self.classes_: the encoder maps
            # class i (encoded) to self._label_encoder.classes_[i], which
            # is exactly self.classes_ above, and XGBoost's predict_proba
            # columns are ordered by the encoded integer classes.
            return self.estimator.predict_proba(X)
        raise NotImplementedError(f"{self.name} does not support predict_proba")


def build_classifier(
    name: str,
    seed: int,
    logistic_regression_max_iter: int = 1000,
    naive_bayes_variant: str = "gaussian",
) -> ClassifierWrapper:
    """Factory: build a `ClassifierWrapper` for one of the six registered
    classifier names, with sklearn/XGBoost default hyperparameters."""
    if name == "mlp":
        estimator = MLPClassifier(random_state=seed)
    elif name == "svc":
        estimator = SVC(probability=True, random_state=seed)
    elif name == "decision_tree":
        estimator = DecisionTreeClassifier(random_state=seed)
    elif name == "knn":
        estimator = KNeighborsClassifier(kn_neighbors=5)  # sklearn default
    elif name == "logistic_regression":
        # Only tolerated deviation from sklearn defaults, documented per spec:
        # default max_iter=100 frequently fails to converge on BoVW histograms.
        estimator = LogisticRegression(max_iter=logistic_regression_max_iter, random_state=seed)
    elif name == "xgboost":
        if XGBClassifier is None:
            raise ImportError("xgboost is not installed; `pip install xgboost`.")
        estimator = XGBClassifier(random_state=seed, eval_metric="mlogloss")
        # XGBoost's sklearn API requires integer-encoded labels in
        # [0, n_classes); it does not accept string class labels like the
        # other five classifiers do.
        return ClassifierWrapper(name=name, estimator=estimator, needs_label_encoding=True)
    elif name == "naive_bayes":
        # MultinomialNB is the natural fit for count-based visual-word
        # histograms; GaussianNB offered as a configurable alternative.
        estimator = GaussianNB()
    else:
        raise ValueError(f"Unknown classifier name: {name}")

    return ClassifierWrapper(name=name, estimator=estimator)


CLASSIFIER_REGISTRY: Dict[str, str] = {
    "mlp": "MLPClassifier",
    "svc": "SVC",
    "decision_tree": "DecisionTreeClassifier",
    "knn": "KNeighborsClassifier",
    "logistic_regression": "LogisticRegression",
    "xgboost": "XGBClassifier",
    "naive_bayes": "MultinomialNB / GaussianNB",
}
