from __future__ import annotations
from typing import Self, TypeAlias
import numpy as np
from numpy.typing import NDArray
# import math
from py_rust_glm._core import solve_normal_equations, apply_coefficients


DesignMatrix: TypeAlias = NDArray[np.float64]
Target: TypeAlias = NDArray[np.float64]
FitCoefficients: TypeAlias = NDArray[np.float64]
Predictions: TypeAlias = NDArray[np.float64]


class LinearRegression:
    def __init__(self):
        self.coef_: NDArray[np.float64] | None = None
        self.intercept_: float | None = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Self:
        self.coef_ = solve_normal_equations(X, y)
        return self

    def predict(self, X: DesignMatrix) -> Predictions:
        if self.coef_ is None:
            raise RuntimeError("Call fit() before predict()")
        return apply_coefficients(X, self.coef_)



# class LogisticRegression(LinearModel):
#     """
#     Logistic regression via OLS on the log-odds of y.
#     y should be probabilities in (0, 1) or binary {0, 1} labels
#     (binary labels are clipped to avoid log(0)).
#     """

#     def link(self, mu: float) -> float:
#         mu = max(1e-8, min(1 - 1e-8, mu))
#         return math.log(mu / (1 - mu))

#     def inverse_link(self, eta: float) -> float:
#         return 1.0 / (1.0 + math.exp(-eta))

#     def fit(self, x: list[list[float]], y: list[float]) -> "LogisticRegression":
#         y_transformed = [self.link(yi) for yi in y]
#         return super().fit(x, y_transformed)


# class PoissonRegression(LinearModel):
#     """
#     Poisson regression via OLS on log(y).
#     y should be non-negative counts.
#     """

#     def link(self, mu: float) -> float:
#         return math.log(max(mu, 1e-8))

#     def inverse_link(self, eta: float) -> float:
#         return math.exp(eta)

#     def fit(self, x: list[list[float]], y: list[float]) -> "PoissonRegression":
#         y_transformed = [self.link(yi) for yi in y]
#         return super().fit(x, y_transformed)