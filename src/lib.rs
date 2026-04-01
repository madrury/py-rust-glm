#![allow(mixed_script_confusables)]
#![allow(non_snake_case)]
use std::usize;
use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const MAX_ITER: usize = 10;
const TOL: f64 = 1e-8;


#[pyfunction]
fn solve<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = DMatrix::from_row_slice(X.shape()[0], X.shape()[1], X.as_slice().unwrap());
    let y = DVector::from_row_slice(y.as_slice().unwrap());

    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * &y;

    let chol = xtx.cholesky().ok_or_else(|| {
        PyValueError::new_err("XᵀX is not positive definite — check for collinear features")
    })?;
    let β = chol.solve(&xty);
    Ok(PyArray1::from_slice(py, β.as_slice()))
}

#[pyfunction]
fn score<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    β: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let X = DMatrix::from_row_slice(X.shape()[0], X.shape()[1], X.as_slice().unwrap());
    let β = DVector::from_row_slice(β.as_slice().unwrap());
    let y_hat = X * β;
    Ok(PyArray1::from_slice(py, y_hat.as_slice()))
}


#[pyfunction]
fn solvelr<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];
    let X = DMatrix::from_row_slice(
        n_samples,
        n_features,
        X.as_slice().expect("Array must be contiguous"),
    );
    let y = DVector::from_row_slice(y.as_slice().expect("Array must be contiguous"));
    let Xt = X.transpose();

    let mut β: DVector<f64> = DVector::zeros(n_features);
    let mut r: DVector<f64>;
    let mut ν: DVector<f64>;
    let mut p: DVector<f64>;

    for _ in 0..MAX_ITER {
        ν = &X * &β;
        p = ν.map(|νi| 1.0 / (1.0 + (-νi).exp()));
        r = &y - &p;
        let Xtr = &Xt * &r;
        let Xw = DMatrix::from_fn(n_samples, n_features, |i, j| {
            X[(i, j)] * p[i] * (1.0 - p[i])
        });
        let wXtX = Xw.transpose() * &X;
        let chol = wXtX.cholesky().ok_or_else(|| {
            PyValueError::new_err("wXᵀX is not positive definite")
        })?;
        β = β + chol.solve(&Xtr);
    }
    Ok(PyArray1::from_slice(py, β.as_slice()))
}

#[pyfunction]
fn scorelr<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    β: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];
    let X = DMatrix::from_row_slice(
        n_samples,
        n_features,
        X.as_slice().expect("Array must be contiguous"),
    );
    let β = DVector::from_row_slice(β.as_slice().unwrap());
    let ν = X * β;
    let p = ν.map(|νi| 1.0 / (1.0 + (-νi).exp()));
    Ok(PyArray1::from_slice(py, p.as_slice()))
}

#[pyfunction]
fn solvepr<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];
    let X = DMatrix::from_row_slice(
        n_samples,
        n_features,
        X.as_slice().expect("Array must be contiguous"),
    );
    let y = DVector::from_row_slice(y.as_slice().expect("Array must be contiguous"));
    let t = DVector::from_row_slice(t.as_slice().expect("Array must be contiguous"));
    let Xt = X.transpose();

    let mut β: DVector<f64> = DVector::zeros(n_features);
    let mut r: DVector<f64>;
    let mut ν: DVector<f64>;
    let mut λ: DVector<f64>;

    for _ in 0..MAX_ITER {
        ν = &X * &β;
        λ = ν.map(|νi| νi.exp());
        r = DVector::from_fn(n_samples, |i, _| {y[i] - λ[i] * t[i]});
        let Xtr = &Xt * &r;
        let Xw = DMatrix::from_fn(n_samples, n_features, |i, j| {
            X[(i, j)] * λ[i] * t[i]
        });
        let wXtX = Xw.transpose() * &X;
        let chol = wXtX.cholesky().ok_or_else(|| {
            PyValueError::new_err("wXᵀX is not positive definite")
        })?;
        β = β + chol.solve(&Xtr);
    }
    Ok(PyArray1::from_slice(py, β.as_slice()))
}

#[pyfunction]
fn scorepr<'py>(
    py: Python<'py>,
    X: PyReadonlyArray2<'py, f64>,
    t: PyReadonlyArray1<'py, f64>,
    β: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let n_samples = X.shape()[0];
    let n_features = X.shape()[1];
    let X = DMatrix::from_row_slice(
        n_samples,
        n_features,
        X.as_slice().expect("Array must be contiguous"),
    );
    let t = DVector::from_row_slice(t.as_slice().unwrap());
    let β = DVector::from_row_slice(β.as_slice().unwrap());
    let ν = X * β;
    let λ = ν.map(|νi| νi.exp());
    Ok(PyArray1::from_slice(py, (λ * t).as_slice()))
}


#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(solvelr, m)?)?;
    m.add_function(wrap_pyfunction!(scorelr, m)?)?;
    m.add_function(wrap_pyfunction!(solvepr, m)?)?;
    m.add_function(wrap_pyfunction!(scorepr, m)?)?;
    Ok(())
}