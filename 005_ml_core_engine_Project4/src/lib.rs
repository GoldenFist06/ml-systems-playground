use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;


#[pyclass(from_py_object)]
#[derive(Clone, Debug)]
pub struct PyMatrix {
    rows: usize,
    cols: usize,
    data: Vec<f32>,
}

#[pymethods]
impl PyMatrix {
   
    #[new]
    fn new(rows: usize, cols: usize, initial_value: f32) -> Self {
        let data = vec![initial_value; rows * cols];
        PyMatrix { rows, cols, data }
    }

   
    #[staticmethod]
    fn from_list(rows: usize, cols: usize, data: Vec<f32>) -> PyResult<Self> {
        if data.len() != rows * cols {
            return Err(PyValueError::new_err(format!(
                "Data length ({}) does not match dimensions {}x{}={}",
                data.len(), rows, cols, rows * cols
            )));
        }
        Ok(PyMatrix { rows, cols, data })
    }

   
    fn __repr__(&self) -> String {
        format!("PyMatrix(rows={}, cols={}, shape=({}, {}))", self.rows, self.cols, self.rows, self.cols)
    }

    #[getter]
    fn rows(&self) -> usize {
        self.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.cols
    }

    
    fn add(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for addition"));
        }

        let result_data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        Ok(PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        })
    }

   
    fn sub(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for subtraction"));
        }

        let result_data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        Ok(PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        })
    }

  
    fn multiply(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.rows != other.rows || self.cols != other.cols {
            return Err(PyValueError::new_err("Matrix dimensions must match for element-wise multiplication"));
        }

        let result_data: Vec<f32> = self.data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        Ok(PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        })
    }

    
    fn scale(&self, factor: f32) -> PyMatrix {
        let result_data: Vec<f32> = self.data
            .iter()
            .map(|&val| val * factor)
            .collect();

        PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

   
    fn matmul(&self, other: &PyMatrix) -> PyResult<PyMatrix> {
        if self.cols != other.rows {
            return Err(PyValueError::new_err(format!(
                "Cannot multiply matrices of shape ({}, {}) and ({}, {})",
                self.rows, self.cols, other.rows, other.cols
            )));
        }

        let rows = self.rows;
        let cols = other.cols;
        let inner = self.cols;
        
        let mut result_data = vec![0.0; rows * cols];

        for i in 0..rows {
            for k in 0..inner {
                let a_val = self.data[i * inner + k];
                for j in 0..cols {
                    result_data[i * cols + j] += a_val * other.data[k * cols + j];
                }
            }
        }

        Ok(PyMatrix {
            rows,
            cols,
            data: result_data,
        })
    }

  
    fn transpose(&self) -> PyMatrix {
        let new_rows = self.cols;
        let new_cols = self.rows;
        let mut transposed_data = vec![0.0; self.data.len()];

        for r in 0..self.rows {
            for c in 0..self.cols {
                let old_idx = r * self.cols + c;
                let new_idx = c * new_cols + r;
                transposed_data[new_idx] = self.data[old_idx];
            }
        }

        PyMatrix {
            rows: new_rows,
            cols: new_cols,
            data: transposed_data,
        }
    }

   
    fn relu(&self) -> PyMatrix {
        let result_data: Vec<f32> = self.data
            .iter()
            .map(|&val| if val > 0.0 { val } else { 0.0 })
            .collect();

        PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

  
    fn relu_backward(&self) -> PyMatrix {
        let result_data: Vec<f32> = self.data
            .iter()
            .map(|&val| if val > 0.0 { 1.0 } else { 0.0 })
            .collect();

        PyMatrix {
            rows: self.rows,
            cols: self.cols,
            data: result_data,
        }
    }

   
    fn to_list(&self) -> Vec<f32> {
        self.data.clone()
    }
}


#[pymodule]
fn ml_core_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyMatrix>()?;
    Ok(())
}