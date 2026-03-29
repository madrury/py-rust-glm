#![allow(mixed_script_confusables)]
use nalgebra::{DMatrix, DVector};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


#[pyfunction]
fn solve_normal_equations<'py>(
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Vec<f64>> {
    let x = x.as_array();
    let y = y.as_array();

    let x = DMatrix::from_row_slice(x.nrows(), x.ncols(), x.as_slice().unwrap());
    let y = DVector::from_row_slice(y.as_slice().unwrap());

    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * &y;

    // Cholesky is fast and stable for positive-definite XᵀX
    let chol = xtx
        .cholesky()
        .ok_or_else(|| PyValueError::new_err(
            "XᵀX is not positive definite — check for collinear features"
        ))?;

    let beta = chol.solve(&xty);
    Ok(beta.data.into())
}

/// Apply coefficients to a design matrix.
/// X is flat row-major (n_samples × n_features), beta has length n_features.
/// Returns predicted values of length n_samples.
#[pyfunction]
fn apply_coefficients<'py>(
    x: PyReadonlyArray2<'py, f64>,
    β: PyReadonlyArray1<'py, f64>,
) -> PyResult<Vec<f64>> {
    let x = x.as_array();
    let β = β.as_array();

    let x = DMatrix::from_row_slice(x.nrows(), x.ncols(), x.as_slice().unwrap());
    let β = DVector::from_row_slice(β.as_slice().unwrap());
    let y_hat = x * β;
    Ok(y_hat.data.into())
}

#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_normal_equations, m)?)?;
    m.add_function(wrap_pyfunction!(apply_coefficients, m)?)?;
    Ok(())
}