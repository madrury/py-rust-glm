#![allow(mixed_script_confusables)]
use std::usize;
use std::io::{self, Write};

use nalgebra::{DMatrix, DVector};
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;


const MAX_ITER: usize = 10;
const TOL: f64 = 1e-8;


#[pyfunction]
fn solve<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    // let x = x.as_array();
    // let y = y.as_array();

    let x = DMatrix::from_row_slice(x.shape()[0], x.shape()[1], x.as_slice().unwrap());
    let y = DVector::from_row_slice(y.as_slice().unwrap());

    let xt = x.transpose();
    let xtx = &xt * &x;
    let xty = &xt * &y;

    let chol = xtx
        .cholesky()
        .ok_or_else(|| PyValueError::new_err(
            "XᵀX is not positive definite — check for collinear features"
        ))?;

    let β = chol.solve(&xty);
    Ok(PyArray1::from_slice(py, β.as_slice()))
}

#[pyfunction]
fn solvelr<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {


    let n_samples = x.shape()[0];
    let n_features = x.shape()[1];

    let x = DMatrix::from_row_slice(
        n_samples, 
        n_features, 
        x.as_slice().expect("Array must be contiguous")
    );
    let y = DVector::from_row_slice(
        y.as_slice().expect("Array must be contiguous")
    );
    let xt = x.transpose();

    let mut β: DVector<f64> = DVector::zeros(n_features);
    let mut r: DVector<f64>;
    let mut ν : DVector<f64>;
    let mut p: DVector<f64>;
    
    println!("Starting loop.");
    for i in 0..MAX_ITER {
        println!("iter {}: β = {:?}", i, β);
        std::io::stdout().flush().unwrap();
        ν = &x * &β;
        p = ν.map(|νi| 1.0 / (1.0 + (-νi).exp()));
        r = &y - &p;
        let xtr = &xt * &r;
        let xw = DMatrix::from_fn(
            n_samples, 
            n_features, 
            |i, j| x[(i, j)] * p[i] * (1.0 - p[i])
        );
        let wxtx = xw.transpose() * &x;
        let chol = wxtx
            .cholesky()
            .ok_or_else(|| PyValueError::new_err(
                "wXᵀX is not positive definite — check for collinear features"
            ))?;
        β = β + chol.solve(&xtr);
    }
    Ok(PyArray1::from_slice(py, β.as_slice()))
}


#[pyfunction]
fn score<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<'py, f64>,
    β: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let β = β.as_array();

    let x = DMatrix::from_row_slice(x.nrows(), x.ncols(), x.as_slice().unwrap());
    let β = DVector::from_row_slice(β.as_slice().unwrap());
    let y_hat = x * β;
    Ok(PyArray1::from_slice(py, y_hat.as_slice()))
}


#[pymodule]
fn _core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve, m)?)?;
    m.add_function(wrap_pyfunction!(score, m)?)?;
    m.add_function(wrap_pyfunction!(solvelr, m)?)?;
    Ok(())
}