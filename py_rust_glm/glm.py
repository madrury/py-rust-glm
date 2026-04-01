from __future__ import annotations
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

from py_rust_glm._core import solve, solvelr, score, scorelr, solvepr, scorepr


DesignMatrix: TypeAlias = NDArray[np.float64]
Target: TypeAlias = NDArray[np.float64]
Exposure: TypeAlias = NDArray[np.float64]
Coefficients: TypeAlias = NDArray[np.float64]
Predictions: TypeAlias = NDArray[np.float64]


class LinearRegression:

    def __init__(self):
        self.coef_: Coefficients | None = None

    def fit(self, X: DesignMatrix, y: Target) -> Self:
        self.coef_ = solve(X, y)
        return self

    def predict(self, X: DesignMatrix) -> Predictions:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return score(X, self.coef_)


class LogisticRegression:

    def __init__(self):
        self.coef_: Coefficients | None = None

    def fit(self, X: DesignMatrix, y: Target) -> Self:
        self.coef_ = solvelr(X, y)
        return self

    def predict(self, X: DesignMatrix) -> Predictions:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return scorelr(X, self.coef_)


class PoissonRegression:

    def __init__(self):
        self.coef_: Coefficients | None = None

    def fit(self, X: DesignMatrix, y: Target, t: Exposure | None = None) -> Self:
        if t is None:
            t = np.ones(shape=X.shape[0], dtype=np.float64)
        self.coef_ = solvepr(X, y, t)
        return self

    def predict(self, X: DesignMatrix, t: Exposure | None = None) -> Predictions:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        if t is None:
            t = np.ones(shape=X.shape[0], dtype=np.float64)
        return scorepr(X, self.coef_, t)