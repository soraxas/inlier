#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion, clippy::redundant_closure)]

use pyo3::{
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
    wrap_pyfunction,
};

use crate::{
    core::{
        Estimator, InlierSelector, LocalOptimizer, Sampler, Scoring, SuperRansac,
        TerminationCriterion,
    },
    settings::RansacSettings,
    types::DataMatrix,
};

fn matrix_from_python(obj: &Bound<'_, PyAny>) -> PyResult<DataMatrix> {
    let rows: Vec<Vec<f64>> = obj.extract()?;
    if rows.is_empty() {
        return Ok(DataMatrix::from_row_slice(0, 0, &[]));
    }

    let width = rows[0].len();
    if !rows.iter().all(|r| r.len() == width) {
        return Err(PyValueError::new_err("all rows must have the same length"));
    }

    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    Ok(DataMatrix::from_row_slice(flat.len() / width, width, &flat))
}

fn matrix_to_python<'py>(py: Python<'py>, matrix: &DataMatrix) -> PyResult<Bound<'py, PyList>> {
    let rows: Vec<Vec<f64>> = matrix
        .row_iter()
        .map(|row| row.iter().copied().collect::<Vec<f64>>())
        .collect();
    Ok(PyList::new_bound(py, rows))
}

fn indices_to_python<'py>(py: Python<'py>, indices: &[usize]) -> Bound<'py, PyList> {
    PyList::new_bound(py, indices)
}

#[derive(Debug)]
pub struct PyModel {
    inner: Py<PyAny>,
}

impl Clone for PyModel {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

impl From<Py<PyAny>> for PyModel {
    fn from(inner: Py<PyAny>) -> Self {
        Self { inner }
    }
}

impl ToPyObject for PyModel {
    fn to_object(&self, py: Python<'_>) -> PyObject {
        self.inner.clone_ref(py).into_py(py)
    }
}

#[pyclass(name = "RansacSettings")]
#[derive(Clone)]
pub struct PyRansacSettings {
    inner: RansacSettings,
}

#[pymethods]
impl PyRansacSettings {
    #[new]
    #[pyo3(signature = (
        min_iterations=1000,
        max_iterations=5000,
        inlier_threshold=1.5,
        confidence=0.99,
    ))]
    pub fn new(
        min_iterations: usize,
        max_iterations: usize,
        inlier_threshold: f64,
        confidence: f64,
    ) -> Self {
        let settings = RansacSettings {
            min_iterations,
            max_iterations,
            inlier_threshold,
            confidence,
            ..RansacSettings::default()
        };
        Self { inner: settings }
    }

    #[getter]
    pub fn inlier_threshold(&self) -> f64 {
        self.inner.inlier_threshold
    }
}

#[pyclass(name = "EstimatorAdapter")]
pub struct PyEstimatorAdapter {
    inner: Py<PyAny>,
}

impl Clone for PyEstimatorAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyEstimatorAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl Estimator for PyEstimatorAdapter {
    type Model = PyModel;

    fn sample_size(&self) -> usize {
        Python::with_gil(|py| {
            self.inner
                .bind(py)
                .call_method0("sample_size")
                .and_then(|v| v.extract())
                .unwrap_or(0)
        })
    }

    fn non_minimal_sample_size(&self) -> usize {
        Python::with_gil(|py| {
            self.inner
                .bind(py)
                .call_method0("non_minimal_sample_size")
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| self.sample_size())
        })
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        Python::with_gil(|py| {
            let sample_py = indices_to_python(py, sample);
            let data_py = matrix_to_python(py, data).unwrap();
            self.inner
                .bind(py)
                .call_method("is_valid_sample", (data_py, sample_py), None)
                .and_then(|v| v.extract())
                .unwrap_or(false)
        })
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        Python::with_gil(|py| {
            let sample_py = indices_to_python(py, sample);
            let data_py = matrix_to_python(py, data).unwrap();
            self.inner
                .bind(py)
                .call_method("estimate_model", (data_py, sample_py), None)
                .and_then(|v| v.extract::<Vec<Py<PyAny>>>())
                .unwrap_or_default()
                .into_iter()
                .map(PyModel::from)
                .collect()
        })
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        data: &DataMatrix,
        sample: &[usize],
        threshold: f64,
    ) -> bool {
        Python::with_gil(|py| {
            let sample_py = indices_to_python(py, sample);
            let data_py = matrix_to_python(py, data).unwrap();
            self.inner
                .bind(py)
                .call_method(
                    "is_valid_model",
                    (model.inner.clone_ref(py), data_py, sample_py, threshold),
                    None,
                )
                .and_then(|v| v.extract())
                .unwrap_or(true)
        })
    }
}

#[pyclass(name = "ScoringAdapter")]
pub struct PyScoringAdapter {
    inner: Py<PyAny>,
}

impl Clone for PyScoringAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyScoringAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl Scoring<PyModel> for PyScoringAdapter {
    type Score = f64;

    fn threshold(&self) -> f64 {
        Python::with_gil(|py| {
            self.inner
                .bind(py)
                .call_method0("threshold")
                .and_then(|v| v.extract())
                .unwrap_or(0.0)
        })
    }

    fn score(
        &self,
        data: &DataMatrix,
        model: &PyModel,
        inliers_out: &mut Vec<usize>,
    ) -> Self::Score {
        Python::with_gil(|py| {
            let data_py = matrix_to_python(py, data).unwrap();
            let result = self.inner.bind(py).call_method(
                "score",
                (data_py, model.inner.clone_ref(py)),
                None,
            );
            match result.and_then(|v| v.extract::<(f64, Option<Vec<usize>>)>()) {
                Ok((score, maybe_inliers)) => {
                    if let Some(inliers) = maybe_inliers {
                        *inliers_out = inliers;
                    }
                    score
                }
                Err(_) => 0.0,
            }
        })
    }
}

#[pyclass(name = "SamplerAdapter")]
pub struct PySamplerAdapter {
    inner: Py<PyAny>,
}

impl Clone for PySamplerAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PySamplerAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl Sampler for PySamplerAdapter {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        Python::with_gil(|py| {
            let data_py = matrix_to_python(py, data).unwrap();
            let res = self
                .inner
                .bind(py)
                .call_method("sample", (data_py, sample_size), None);
            match res.and_then(|v| v.extract::<Vec<usize>>()) {
                Ok(vec) if vec.len() == sample_size => {
                    out_indices.copy_from_slice(&vec);
                    true
                }
                _ => false,
            }
        })
    }

    fn update(&mut self, sample: &[usize], sample_size: usize, iteration: usize, score_hint: f64) {
        Python::with_gil(|py| {
            let _ = self.inner.bind(py).call_method(
                "update",
                (
                    indices_to_python(py, sample),
                    sample_size,
                    iteration,
                    score_hint,
                ),
                None,
            );
        });
    }
}

#[pyclass(name = "LocalOptimizerAdapter")]
pub struct PyLocalOptimizerAdapter {
    inner: Py<PyAny>,
}

impl Clone for PyLocalOptimizerAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyLocalOptimizerAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl LocalOptimizer<PyModel, f64> for PyLocalOptimizerAdapter {
    fn run(
        &mut self,
        data: &DataMatrix,
        inliers: &[usize],
        model: &PyModel,
        best_score: &f64,
    ) -> (PyModel, f64, Vec<usize>) {
        Python::with_gil(|py| {
            let data_py = matrix_to_python(py, data).unwrap();
            let inliers_py = indices_to_python(py, inliers);
            let res = self.inner.bind(py).call_method(
                "run",
                (data_py, inliers_py, model.inner.clone_ref(py), *best_score),
                None,
            );
            res.and_then(|v| v.extract::<(Py<PyAny>, f64, Vec<usize>)>())
                .map(|(m, s, idx)| (PyModel::from(m), s, idx))
                .unwrap_or((model.clone(), *best_score, inliers.to_vec()))
        })
    }
}

#[pyclass(name = "TerminationAdapter")]
pub struct PyTerminationAdapter {
    inner: Py<PyAny>,
}

impl Clone for PyTerminationAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyTerminationAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl TerminationCriterion<f64> for PyTerminationAdapter {
    fn check(
        &mut self,
        data: &DataMatrix,
        best_score: &f64,
        sample_size: usize,
        max_iterations: &mut usize,
    ) -> bool {
        Python::with_gil(|py| {
            let data_py = matrix_to_python(py, data).unwrap();
            let result = self.inner.bind(py).call_method(
                "check",
                (data_py, *best_score, sample_size, *max_iterations),
                None,
            );
            match result.and_then(|v| v.extract::<(bool, Option<usize>)>()) {
                Ok((terminate, maybe_max)) => {
                    if let Some(m) = maybe_max {
                        *max_iterations = m;
                    }
                    terminate
                }
                Err(_) => false,
            }
        })
    }
}

#[pyclass(name = "InlierSelectorAdapter")]
pub struct PyInlierSelectorAdapter {
    inner: Py<PyAny>,
}

impl Clone for PyInlierSelectorAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
        })
    }
}

#[pymethods]
impl PyInlierSelectorAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
        }
    }
}

impl InlierSelector<PyModel> for PyInlierSelectorAdapter {
    fn select(&mut self, data: &DataMatrix, model: &PyModel) -> Vec<usize> {
        Python::with_gil(|py| {
            let data_py = matrix_to_python(py, data).unwrap();
            self.inner
                .bind(py)
                .call_method("select", (data_py, model.inner.clone_ref(py)), None)
                .and_then(|v| v.extract())
                .unwrap_or_else(|_| (0..data.nrows()).collect())
        })
    }
}

fn matrix3_to_python<'py>(
    py: Python<'py>,
    m: nalgebra::Matrix3<f64>,
) -> PyResult<Bound<'py, PyList>> {
    let rows = vec![
        vec![m[(0, 0)], m[(0, 1)], m[(0, 2)]],
        vec![m[(1, 0)], m[(1, 1)], m[(1, 2)]],
        vec![m[(2, 0)], m[(2, 1)], m[(2, 2)]],
    ];
    Ok(PyList::new_bound(py, rows))
}

fn vec3_to_python<'py>(
    py: Python<'py>,
    v: &nalgebra::Vector3<f64>,
) -> PyResult<Bound<'py, PyList>> {
    Ok(PyList::new_bound(py, [v[0], v[1], v[2]]))
}

#[pyfunction(signature = (points1, points2, threshold=1.5, settings=None))]
pub fn estimate_homography_py(
    points1: Bound<PyAny>,
    points2: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points1.py();
    let data1 = matrix_from_python(&points1)?;
    let data2 = matrix_from_python(&points2)?;
    let result = crate::estimate_homography(&data1, &data2, threshold, settings.map(|s| s.inner))
        .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item("model", matrix3_to_python(py, result.model.h)?)?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points1, points2, threshold=1.5, settings=None))]
pub fn estimate_fundamental_matrix_py(
    points1: Bound<PyAny>,
    points2: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points1.py();
    let data1 = matrix_from_python(&points1)?;
    let data2 = matrix_from_python(&points2)?;
    let result =
        crate::estimate_fundamental_matrix(&data1, &data2, threshold, settings.map(|s| s.inner))
            .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item("model", matrix3_to_python(py, result.model.f)?)?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points1, points2, threshold=1.5, settings=None))]
pub fn estimate_essential_matrix_py(
    points1: Bound<PyAny>,
    points2: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points1.py();
    let data1 = matrix_from_python(&points1)?;
    let data2 = matrix_from_python(&points2)?;
    let result =
        crate::estimate_essential_matrix(&data1, &data2, threshold, settings.map(|s| s.inner))
            .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item("model", matrix3_to_python(py, result.model.e)?)?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points_3d, points_2d, threshold=1.5, settings=None))]
pub fn estimate_absolute_pose_py(
    points_3d: Bound<PyAny>,
    points_2d: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points_3d.py();
    let data_3d = matrix_from_python(&points_3d)?;
    let data_2d = matrix_from_python(&points_2d)?;
    let result =
        crate::estimate_absolute_pose(&data_3d, &data_2d, threshold, settings.map(|s| s.inner))
            .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item(
        "rotation",
        matrix3_to_python(py, *result.model.rotation.to_rotation_matrix().matrix())?,
    )?;
    out.set_item(
        "translation",
        vec3_to_python(py, &result.model.translation.vector)?,
    )?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points1, points2, threshold=1.5, settings=None))]
pub fn estimate_rigid_transform_py(
    points1: Bound<PyAny>,
    points2: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points1.py();
    let data1 = matrix_from_python(&points1)?;
    let data2 = matrix_from_python(&points2)?;
    let result =
        crate::estimate_rigid_transform(&data1, &data2, threshold, settings.map(|s| s.inner))
            .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item(
        "rotation",
        matrix3_to_python(py, *result.model.rotation.to_rotation_matrix().matrix())?,
    )?;
    out.set_item(
        "translation",
        vec3_to_python(py, &result.model.translation.vector)?,
    )?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction(signature = (points, threshold=1.5, settings=None))]
pub fn estimate_line_py(
    points: Bound<PyAny>,
    threshold: f64,
    settings: Option<PyRansacSettings>,
) -> PyResult<Py<PyDict>> {
    let py = points.py();
    let data = matrix_from_python(&points)?;
    let result = crate::estimate_line(&data, threshold, settings.map(|s| s.inner))
        .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item("model", vec3_to_python(py, result.model.params())?)?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
}

#[pyfunction]
pub fn probe_estimator(
    estimator: PyEstimatorAdapter,
    sample: Vec<usize>,
    data: Bound<PyAny>,
) -> PyResult<usize> {
    let matrix = matrix_from_python(&data)?;
    let valid = estimator.is_valid_sample(&matrix, &sample);
    Ok(if valid { estimator.sample_size() } else { 0 })
}

#[allow(clippy::too_many_arguments)]
#[pyfunction(signature = (
    estimator,
    sampler,
    scoring,
    local_optimizer=None,
    termination=None,
    inlier_selector=None,
    settings=None,
    data=None,
))]
pub fn run_python_ransac(
    estimator: PyEstimatorAdapter,
    sampler: PySamplerAdapter,
    scoring: PyScoringAdapter,
    local_optimizer: Option<PyLocalOptimizerAdapter>,
    termination: Option<PyTerminationAdapter>,
    mut inlier_selector: Option<PyInlierSelectorAdapter>,
    settings: Option<PyRansacSettings>,
    data: Option<Bound<PyAny>>,
) -> PyResult<Option<(PyObject, Vec<usize>, f64)>> {
    let data = data.ok_or_else(|| PyValueError::new_err("data is required"))?;
    let matrix = matrix_from_python(&data)?;
    let settings = settings.map(|s| s.inner).unwrap_or_default();
    let mut pipeline = SuperRansac::new(
        settings,
        estimator,
        sampler,
        scoring,
        local_optimizer,
        None,
        termination.unwrap_or_else(|| PyTerminationAdapter {
            inner: Python::with_gil(|py| py.None()),
        }),
        inlier_selector.take(),
    );
    pipeline.run(&matrix);

    Python::with_gil(|py| {
        Ok(pipeline.best_model.map(|model| {
            (
                model.to_object(py),
                pipeline.best_inliers,
                pipeline.best_score.unwrap_or(0.0),
            )
        }))
    })
}

#[pymodule]
fn _inlier_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEstimatorAdapter>()?;
    m.add_class::<PyScoringAdapter>()?;
    m.add_class::<PySamplerAdapter>()?;
    m.add_class::<PyLocalOptimizerAdapter>()?;
    m.add_class::<PyTerminationAdapter>()?;
    m.add_class::<PyInlierSelectorAdapter>()?;
    m.add_class::<PyRansacSettings>()?;

    m.add_function(wrap_pyfunction!(estimate_homography_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_fundamental_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_essential_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_absolute_pose_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rigid_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_line_py, m)?)?;
    m.add_function(wrap_pyfunction!(probe_estimator, m)?)?;
    m.add_function(wrap_pyfunction!(run_python_ransac, m)?)?;

    let _ = py;
    Ok(())
}
