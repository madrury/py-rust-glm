from __future__ import annotations
from typing import Self, TypeAlias

import numpy as np
from numpy.typing import NDArray

from py_rust_glm._core import solve, score


DesignMatrix: TypeAlias = NDArray[np.float64]
Target: TypeAlias = NDArray[np.float64]
FitCoefficients: TypeAlias = NDArray[np.float64]
Predictions: TypeAlias = NDArray[np.float64]


class LinearRegression:

    def __init__(self):
        self.coef_: NDArray[np.float64] | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        self.coef_ = solve(X, y)
        return self

    def predict(self, X: DesignMatrix) -> Predictions:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return score(X, self.coef_)