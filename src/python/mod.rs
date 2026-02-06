#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::useless_conversion, clippy::redundant_closure)]

use pyo3::{
    FromPyObject,
    exceptions::PyValueError,
    prelude::*,
    types::{PyDict, PyList},
    wrap_pyfunction,
};

use crate::{
    core::{
        Estimator, InlierSelector, LocalOptimizer, MetaSAC, Sampler, Scoring, TerminationCriterion,
    },
    settings::MetasacSettings,
    types::DataMatrix,
};
use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use std::sync::Arc;

/// Rust-owned data matrix that can be reused without copying each call.
#[pyclass(name = "DataMatrix")]
pub struct PyDataMatrix {
    pub(crate) inner: Arc<DataMatrix>,
}

#[pymethods]
impl PyDataMatrix {
    #[new]
    #[pyo3(signature = (data))]
    pub fn new(data: Bound<PyAny>) -> PyResult<Self> {
        let inner = matrix_from_python_owned(&data)?;
        Ok(Self {
            inner: Arc::new(inner),
        })
    }

    fn __len__(&self) -> usize {
        self.inner.nrows()
    }

    fn __getitem__(&self, idx: usize) -> Option<Vec<f64>> {
        if idx >= self.inner.nrows() {
            return None;
        }
        let mut row = Vec::with_capacity(self.inner.ncols());
        for c in 0..self.inner.ncols() {
            row.push(self.inner[(idx, c)]);
        }
        Some(row)
    }
}

/// Input matrix that either borrows from a PyDataMatrix or owns its own copy.
enum MatrixInput<'py> {
    Borrowed(PyRef<'py, PyDataMatrix>),
    Owned {
        data: Arc<DataMatrix>,
        handle: Py<PyAny>,
    },
}

impl<'py> MatrixInput<'py> {
    fn as_matrix(&self) -> &DataMatrix {
        match self {
            MatrixInput::Borrowed(r) => &r.inner,
            MatrixInput::Owned { data, .. } => data,
        }
    }

    fn py_handle(&self, py: Python<'py>) -> PyResult<Py<PyAny>> {
        match self {
            MatrixInput::Borrowed(r) => {
                let rows = r.inner.nrows();
                let cols = r.inner.ncols();
                let mut data = Vec::with_capacity(rows * cols);
                for rr in 0..rows {
                    let mut row = Vec::with_capacity(cols);
                    for c in 0..cols {
                        row.push(r.inner[(rr, c)]);
                    }
                    data.push(row);
                }
                PyArray2::from_vec2_bound(py, &data)
                    .map(|arr| arr.into_py(py))
                    .map_err(|e| PyValueError::new_err(format!("failed to create numpy view: {e}")))
            }
            MatrixInput::Owned { data, handle: _ } => {
                let rows = data.nrows();
                let cols = data.ncols();
                let mut data_vec = Vec::with_capacity(rows);
                for r in 0..rows {
                    let mut row = Vec::with_capacity(cols);
                    for c in 0..cols {
                        row.push(data[(r, c)]);
                    }
                    data_vec.push(row);
                }
                PyArray2::from_vec2_bound(py, &data_vec)
                    .map(|arr| arr.into_py(py))
                    .map_err(|e| PyValueError::new_err(format!("failed to create numpy view: {e}")))
            }
        }
    }
}

/// Convert Python input into a Rust-owned DataMatrix (one-time copy).
fn matrix_from_python_owned(obj: &Bound<'_, PyAny>) -> PyResult<DataMatrix> {
    {
        if let Ok(arr) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
            let shape = arr.shape();
            let rows = shape[0];
            let cols = shape[1];
            let slice = arr.as_slice()?;
            return Ok(DataMatrix::from_row_slice(rows, cols, slice));
        }
    }

    // Simple path: list-of-lists (copies).
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

/// Convert Python input into a borrowed-or-owned matrix wrapper.
fn matrix_input_from_pyany<'py>(obj: Bound<'py, PyAny>) -> PyResult<MatrixInput<'py>> {
    if let Ok(dm) = obj.extract::<PyRef<PyDataMatrix>>() {
        return Ok(MatrixInput::Borrowed(dm));
    }
    let data = Arc::new(matrix_from_python_owned(&obj)?);
    Python::with_gil(|py| {
        // Build a Python view once for callbacks.
        let rows = data.nrows();
        let cols = data.ncols();
        let mut data_vec = Vec::with_capacity(rows);
        for r in 0..rows {
            let mut row = Vec::with_capacity(cols);
            for c in 0..cols {
                row.push(data[(r, c)]);
            }
            data_vec.push(row);
        }
        let arr = PyArray2::from_vec2_bound(py, &data_vec)
            .map_err(|e| PyValueError::new_err(format!("failed to create numpy array: {e}")))?;
        Ok(MatrixInput::Owned {
            data,
            handle: arr.into_py(py),
        })
    })
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

fn data_to_pyobject<'py>(
    py: Python<'py>,
    matrix: &DataMatrix,
    handle: Option<&Py<PyAny>>,
) -> PyObject {
    match handle {
        Some(h) => h.clone_ref(py).into_py(py),
        None => {
            let rows = matrix.nrows();
            let cols = matrix.ncols();
            let mut data_vec = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row = Vec::with_capacity(cols);
                for c in 0..cols {
                    row.push(matrix[(r, c)]);
                }
                data_vec.push(row);
            }
            PyArray2::from_vec2_bound(py, &data_vec)
                .map(|arr| arr.into_py(py))
                .unwrap_or_else(|_| py.None())
        }
    }
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

#[pyclass(name = "MetasacSettings")]
#[derive(Clone)]
pub struct PyMetasacSettings {
    inner: MetasacSettings,
}

#[pymethods]
impl PyMetasacSettings {
    #[new]
    #[pyo3(signature = (
        min_iterations=1000,
        max_iterations=5000,
        inlier_threshold=1.5,
        confidence=0.99,
        rng_seed=None,
        sampler="uniform",
    ))]
    pub fn new(
        min_iterations: usize,
        max_iterations: usize,
        inlier_threshold: f64,
        confidence: f64,
        rng_seed: Option<u64>,
        sampler: &str,
    ) -> Self {
        let sampler = match sampler.to_ascii_lowercase().as_str() {
            "uniform" => crate::settings::SamplerType::Uniform,
            _ => crate::settings::SamplerType::Prosac,
        };
        let settings = MetasacSettings {
            min_iterations,
            max_iterations,
            inlier_threshold,
            confidence,
            rng_seed,
            sampler,
            ..MetasacSettings::default()
        };
        Self { inner: settings }
    }

    #[getter]
    pub fn inlier_threshold(&self) -> f64 {
        self.inner.inlier_threshold
    }

    #[getter]
    pub fn rng_seed(&self) -> Option<u64> {
        self.inner.rng_seed
    }

    /// Set the sampler type. Accepted values: "uniform" (default) or "prosac".
    pub fn set_sampler(&mut self, sampler: &str) -> PyResult<()> {
        match sampler.to_ascii_lowercase().as_str() {
            "uniform" | "uniform_random" => {
                self.inner.sampler = crate::settings::SamplerType::Uniform;
                Ok(())
            }
            "prosac" => {
                self.inner.sampler = crate::settings::SamplerType::Prosac;
                Ok(())
            }
            other => Err(PyValueError::new_err(format!(
                "unknown sampler '{other}', expected 'uniform' or 'prosac'"
            ))),
        }
    }
}

/// Simple Python-side pipeline builder that wires the components and runs RANSAC.
#[pyclass(name = "Pipeline")]
#[derive(Clone)]
pub struct PyPipeline {
    estimator: PyEstimatorHandle,
    sampler: PySamplerEither,
    scoring: PyScoringAdapter,
    local_optimizer: Option<PyLocalOptimizerAdapter>,
    termination: Option<PyTerminationAdapter>,
    inlier_selector: Option<PyInlierSelectorAdapter>,
    settings: Option<PyMetasacSettings>,
}

#[pymethods]
impl PyPipeline {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        estimator,
        sampler,
        scoring,
        local_optimizer=None,
        termination=None,
        inlier_selector=None,
        settings=None,
    ))]
    pub fn new(
        estimator: PyEstimatorHandle,
        sampler: PySamplerEither,
        scoring: PyScoringAdapter,
        local_optimizer: Option<PyLocalOptimizerAdapter>,
        termination: Option<PyTerminationAdapter>,
        inlier_selector: Option<PyInlierSelectorAdapter>,
        settings: Option<PyMetasacSettings>,
    ) -> Self {
        Self {
            estimator,
            sampler,
            scoring,
            local_optimizer,
            termination,
            inlier_selector,
            settings,
        }
    }

    // TODO: can we take ownership of the estimator, and avoid the clone?
    pub fn with_estimator(&self, estimator: PyEstimatorHandle) -> Self {
        Self {
            estimator,
            ..self.clone()
        }
    }

    pub fn with_sampler(&self, sampler: PySamplerEither) -> Self {
        Self {
            sampler,
            ..self.clone()
        }
    }

    pub fn with_scoring(&self, scoring: PyScoringAdapter) -> Self {
        Self {
            scoring,
            ..self.clone()
        }
    }

    #[pyo3(signature = (local_optimizer))]
    pub fn with_local_optimizer(&self, local_optimizer: Option<PyLocalOptimizerAdapter>) -> Self {
        Self {
            local_optimizer,
            ..self.clone()
        }
    }

    #[pyo3(signature = (termination))]
    pub fn with_termination(&self, termination: Option<PyTerminationAdapter>) -> Self {
        Self {
            termination,
            ..self.clone()
        }
    }

    #[pyo3(signature = (inlier_selector))]
    pub fn with_inlier_selector(&self, inlier_selector: Option<PyInlierSelectorAdapter>) -> Self {
        Self {
            inlier_selector,
            ..self.clone()
        }
    }

    #[pyo3(signature = (settings))]
    pub fn with_settings(&self, settings: Option<PyMetasacSettings>) -> Self {
        Self {
            settings,
            ..self.clone()
        }
    }

    /// Run the configured pipeline on the given data matrix.
    #[pyo3(signature = (data))]
    pub fn run(&self, data: Bound<PyAny>) -> PyResult<Option<(PyObject, Vec<usize>, f64)>> {
        run_metasac(
            self.estimator.clone(),
            self.sampler.clone(),
            self.scoring.clone(),
            self.local_optimizer.clone(),
            self.termination.clone(),
            self.inlier_selector.clone(),
            self.settings.clone(),
            Some(data),
        )
    }
}

#[pyclass(name = "EstimatorAdapter")]
pub struct PyEstimatorAdapter {
    inner: Py<PyAny>,
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PyEstimatorAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyEstimatorAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PyEstimatorAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
    }
}

#[pyclass(name = "HomographyEstimator")]
#[derive(Clone)]
pub struct PyHomographyEstimator;

#[pyclass(name = "FundamentalEstimator")]
#[derive(Clone)]
pub struct PyFundamentalEstimator;

#[pyclass(name = "EssentialEstimator")]
#[derive(Clone)]
pub struct PyEssentialEstimator;

#[pyclass(name = "AbsolutePoseEstimator")]
#[derive(Clone)]
pub struct PyAbsolutePoseEstimator;

#[pyclass(name = "RigidTransformEstimator")]
#[derive(Clone)]
pub struct PyRigidTransformEstimator;

#[pyclass(name = "LineEstimatorNative")]
#[derive(Clone)]
pub struct PyLineEstimatorNative;

/// A single estimator handle that can be backed by a Python adapter or a native estimator.
#[derive(Clone, FromPyObject)]
pub enum PyEstimatorHandle {
    #[pyo3(annotation = "EstimatorAdapter")]
    Py(PyEstimatorAdapter),
    #[pyo3(annotation = "HomographyEstimator")]
    Homography(PyHomographyEstimator),
    #[pyo3(annotation = "FundamentalEstimator")]
    Fundamental(PyFundamentalEstimator),
    #[pyo3(annotation = "EssentialEstimator")]
    Essential(PyEssentialEstimator),
    #[pyo3(annotation = "AbsolutePoseEstimator")]
    AbsolutePose(PyAbsolutePoseEstimator),
    #[pyo3(annotation = "RigidTransformEstimator")]
    RigidTransform(PyRigidTransformEstimator),
    #[pyo3(annotation = "LineEstimatorNative")]
    Line(PyLineEstimatorNative),
}

impl PyEstimatorHandle {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        match self {
            PyEstimatorHandle::Py(e) => PyEstimatorHandle::Py(e.with_data_handle(handle)),
            other => other.clone(),
        }
    }
}

impl Estimator for PyEstimatorHandle {
    type Model = PyModel;

    fn sample_size(&self) -> usize {
        match self {
            PyEstimatorHandle::Py(e) => e.sample_size(),
            PyEstimatorHandle::Homography(_) => {
                crate::estimators::HomographyEstimator::new().sample_size()
            }
            PyEstimatorHandle::Fundamental(_) => {
                crate::estimators::FundamentalEstimator::new().sample_size()
            }
            PyEstimatorHandle::Essential(_) => {
                crate::estimators::EssentialEstimator::new().sample_size()
            }
            PyEstimatorHandle::AbsolutePose(_) => {
                crate::estimators::AbsolutePoseEstimator::new().sample_size()
            }
            PyEstimatorHandle::RigidTransform(_) => {
                crate::estimators::RigidTransformEstimator::new().sample_size()
            }
            PyEstimatorHandle::Line(_) => crate::estimators::LineEstimator::new().sample_size(),
        }
    }

    fn non_minimal_sample_size(&self) -> usize {
        match self {
            PyEstimatorHandle::Py(e) => e.non_minimal_sample_size(),
            _ => self.sample_size(),
        }
    }

    fn is_valid_sample(&self, data: &DataMatrix, sample: &[usize]) -> bool {
        match self {
            PyEstimatorHandle::Py(e) => e.is_valid_sample(data, sample),
            PyEstimatorHandle::Homography(_) => {
                crate::estimators::HomographyEstimator::new().is_valid_sample(data, sample)
            }
            PyEstimatorHandle::Fundamental(_) => {
                crate::estimators::FundamentalEstimator::new().is_valid_sample(data, sample)
            }
            PyEstimatorHandle::Essential(_) => {
                crate::estimators::EssentialEstimator::new().is_valid_sample(data, sample)
            }
            PyEstimatorHandle::AbsolutePose(_) => {
                crate::estimators::AbsolutePoseEstimator::new().is_valid_sample(data, sample)
            }
            PyEstimatorHandle::RigidTransform(_) => {
                crate::estimators::RigidTransformEstimator::new().is_valid_sample(data, sample)
            }
            PyEstimatorHandle::Line(_) => {
                crate::estimators::LineEstimator::new().is_valid_sample(data, sample)
            }
        }
    }

    fn estimate_model(&self, data: &DataMatrix, sample: &[usize]) -> Vec<Self::Model> {
        match self {
            PyEstimatorHandle::Py(e) => e.estimate_model(data, sample),
            PyEstimatorHandle::Homography(_) => {
                let models =
                    crate::estimators::HomographyEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| matrix3_to_python(py, m.h).ok())
                        .map(|v| PyModel::from(v.into_py(py)))
                        .collect()
                })
            }
            PyEstimatorHandle::Fundamental(_) => {
                let models =
                    crate::estimators::FundamentalEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| matrix3_to_python(py, m.f).ok())
                        .map(|v| PyModel::from(v.into_py(py)))
                        .collect()
                })
            }
            PyEstimatorHandle::Essential(_) => {
                let models =
                    crate::estimators::EssentialEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| matrix3_to_python(py, m.e).ok())
                        .map(|v| PyModel::from(v.into_py(py)))
                        .collect()
                })
            }
            PyEstimatorHandle::AbsolutePose(_) => {
                let models =
                    crate::estimators::AbsolutePoseEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| {
                            let dict = PyDict::new_bound(py);
                            let rot =
                                matrix3_to_python(py, *m.rotation.to_rotation_matrix().matrix())
                                    .ok()?;
                            let t = vec3_to_python(py, &m.translation.vector).ok()?;
                            dict.set_item("rotation", rot).ok()?;
                            dict.set_item("translation", t).ok()?;
                            Some(PyModel::from(dict.into_py(py)))
                        })
                        .collect()
                })
            }
            PyEstimatorHandle::RigidTransform(_) => {
                let models =
                    crate::estimators::RigidTransformEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| {
                            let dict = PyDict::new_bound(py);
                            let rot =
                                matrix3_to_python(py, *m.rotation.to_rotation_matrix().matrix())
                                    .ok()?;
                            let t = vec3_to_python(py, &m.translation.vector).ok()?;
                            dict.set_item("rotation", rot).ok()?;
                            dict.set_item("translation", t).ok()?;
                            Some(PyModel::from(dict.into_py(py)))
                        })
                        .collect()
                })
            }
            PyEstimatorHandle::Line(_) => {
                let models = crate::estimators::LineEstimator::new().estimate_model(data, sample);
                Python::with_gil(|py| {
                    models
                        .into_iter()
                        .filter_map(|m| vec3_to_python(py, m.params()).ok())
                        .map(|v| PyModel::from(v.into_py(py)))
                        .collect()
                })
            }
        }
    }

    fn is_valid_model(
        &self,
        model: &Self::Model,
        data: &DataMatrix,
        sample: &[usize],
        threshold: f64,
    ) -> bool {
        match self {
            PyEstimatorHandle::Py(e) => e.is_valid_model(model, data, sample, threshold),
            _ => true,
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PyScoringAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyScoringAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PyScoringAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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

/// Simple native scoring: inlier count with a fixed threshold.
// Simplify scoring: use Python-defined scoring adapter.

#[pyclass(name = "SamplerAdapter")]
pub struct PySamplerAdapter {
    inner: Py<PyAny>,
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PySamplerAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PySamplerAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PySamplerAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
    }
}

// #[hotpath::measure_all]
impl Sampler for PySamplerAdapter {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        Python::with_gil(|py| {
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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

#[derive(Clone)]
pub enum NativeSampler {
    Uniform,
    Prosac,
}

#[pyclass(name = "NativeSampler")]
#[derive(Clone)]
pub struct PyNativeSampler {
    kind: NativeSampler,
}

#[pymethods]
impl PyNativeSampler {
    #[new]
    pub fn new(kind: &str) -> PyResult<Self> {
        let kind = match kind.to_ascii_lowercase().as_str() {
            "uniform" => NativeSampler::Uniform,
            "prosac" => NativeSampler::Prosac,
            other => return Err(PyValueError::new_err(format!("unknown sampler '{other}'"))),
        };
        Ok(Self { kind })
    }
}

#[derive(Clone, FromPyObject)]
pub enum PySamplerEither {
    #[pyo3(annotation = "SamplerAdapter")]
    Py(PySamplerAdapter),
    #[pyo3(annotation = "NativeSampler")]
    Native(PyNativeSampler),
}

impl PySamplerEither {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        match self {
            PySamplerEither::Py(s) => PySamplerEither::Py(s.with_data_handle(handle)),
            PySamplerEither::Native(n) => PySamplerEither::Native(n.clone()),
        }
    }
}

impl Sampler for PySamplerEither {
    fn sample(&mut self, data: &DataMatrix, sample_size: usize, out_indices: &mut [usize]) -> bool {
        match self {
            PySamplerEither::Py(s) => s.sample(data, sample_size, out_indices),
            PySamplerEither::Native(n) => {
                match n.kind {
                    NativeSampler::Uniform => crate::samplers::UniformRandomSampler::default()
                        .sample(data, sample_size, out_indices),
                    NativeSampler::Prosac => {
                        let mut s = crate::samplers::ProsacSampler::default();
                        s.sample(data, sample_size, out_indices)
                    }
                }
            }
        }
    }

    fn update(&mut self, sample: &[usize], sample_size: usize, iteration: usize, score_hint: f64) {
        match self {
            PySamplerEither::Py(s) => s.update(sample, sample_size, iteration, score_hint),
            PySamplerEither::Native(_) => {
                let _ = sample_size;
                let _ = iteration;
                let _ = score_hint;
            }
        }
    }
}

#[pyclass(name = "LocalOptimizerAdapter")]
pub struct PyLocalOptimizerAdapter {
    inner: Py<PyAny>,
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PyLocalOptimizerAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyLocalOptimizerAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PyLocalOptimizerAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PyTerminationAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyTerminationAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PyTerminationAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
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
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
    data_handle: Option<Py<PyAny>>,
}

impl Clone for PyInlierSelectorAdapter {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            data_handle: self.data_handle.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

#[pymethods]
impl PyInlierSelectorAdapter {
    #[new]
    pub fn new(obj: Bound<PyAny>) -> Self {
        Self {
            inner: obj.unbind(),
            data_handle: None,
        }
    }
}

impl PyInlierSelectorAdapter {
    fn with_data_handle(&self, handle: Py<PyAny>) -> Self {
        let mut cloned = self.clone();
        cloned.data_handle = Some(handle);
        cloned
    }
}

impl InlierSelector<PyModel> for PyInlierSelectorAdapter {
    fn select(&mut self, data: &DataMatrix, model: &PyModel) -> Vec<usize> {
        Python::with_gil(|py| {
            let data_py = data_to_pyobject(py, data, self.data_handle.as_ref());
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points1.py();
    let data1 = matrix_input_from_pyany(points1)?;
    let data2 = matrix_input_from_pyany(points2)?;
    let result = crate::estimate_homography(
        data1.as_matrix(),
        data2.as_matrix(),
        threshold,
        settings.map(|s| s.inner),
    )
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points1.py();
    let data1 = matrix_input_from_pyany(points1)?;
    let data2 = matrix_input_from_pyany(points2)?;
    let result = crate::estimate_fundamental_matrix(
        data1.as_matrix(),
        data2.as_matrix(),
        threshold,
        settings.map(|s| s.inner),
    )
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points1.py();
    let data1 = matrix_input_from_pyany(points1)?;
    let data2 = matrix_input_from_pyany(points2)?;
    let result = crate::estimate_essential_matrix(
        data1.as_matrix(),
        data2.as_matrix(),
        threshold,
        settings.map(|s| s.inner),
    )
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points_3d.py();
    let data_3d = matrix_input_from_pyany(points_3d)?;
    let data_2d = matrix_input_from_pyany(points_2d)?;
    let result = crate::estimate_absolute_pose(
        data_3d.as_matrix(),
        data_2d.as_matrix(),
        threshold,
        settings.map(|s| s.inner),
    )
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points1.py();
    let data1 = matrix_input_from_pyany(points1)?;
    let data2 = matrix_input_from_pyany(points2)?;
    let result = crate::estimate_rigid_transform(
        data1.as_matrix(),
        data2.as_matrix(),
        threshold,
        settings.map(|s| s.inner),
    )
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
    settings: Option<PyMetasacSettings>,
) -> PyResult<Py<PyDict>> {
    // ensure_hotpath_guard(None);
    let py = points.py();
    let data = matrix_input_from_pyany(points)?;
    let result = crate::estimate_line(data.as_matrix(), threshold, settings.map(|s| s.inner))
        .map_err(|e| PyValueError::new_err(e))?;
    let out = PyDict::new_bound(py);
    out.set_item("model", vec3_to_python(py, result.model.params())?)?;
    out.set_item("inliers", result.inliers)?;
    out.set_item("score", result.score.value)?;
    Ok(out.unbind())
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
pub fn run_metasac(
    estimator: PyEstimatorHandle,
    sampler: PySamplerEither,
    scoring: PyScoringAdapter,
    local_optimizer: Option<PyLocalOptimizerAdapter>,
    termination: Option<PyTerminationAdapter>,
    mut inlier_selector: Option<PyInlierSelectorAdapter>,
    settings: Option<PyMetasacSettings>,
    data: Option<Bound<PyAny>>,
) -> PyResult<Option<(PyObject, Vec<usize>, f64)>> {
    let data = data.ok_or_else(|| PyValueError::new_err("data is required"))?;
    let matrix = matrix_input_from_pyany(data)?;
    let data_handle = Python::with_gil(|py| matrix.py_handle(py))?;
    let settings = settings.map(|s| s.inner).unwrap_or_default();
    let estimator = estimator.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py)));
    let sampler = sampler.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py)));
    let scoring = scoring.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py)));
    let local_optimizer = local_optimizer
        .map(|lo| lo.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py))));
    let termination =
        termination.map(|t| t.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py))));
    let inlier_selector = inlier_selector
        .take()
        .map(|sel| sel.with_data_handle(Python::with_gil(|py| data_handle.clone_ref(py))));

    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut pipeline = MetaSAC::new(
            settings,
            estimator,
            sampler,
            scoring,
            local_optimizer,
            None,
            termination.unwrap_or_else(|| PyTerminationAdapter {
                inner: Python::with_gil(|py| py.None()),
                data_handle: None,
            }),
            inlier_selector,
        );
        pipeline.run(matrix.as_matrix());
        Python::with_gil(|py| {
            pipeline.best_model.map(|model| {
                (
                    model.to_object(py),
                    pipeline.best_inliers,
                    pipeline.best_score.unwrap_or(0.0),
                )
            })
        })
    }));

    match result {
        Ok(res) => Ok(res),
        Err(_) => Err(PyValueError::new_err("RANSAC run panicked")),
    }
}

#[pymodule]
fn _inlier_rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEstimatorAdapter>()?;
    m.add_class::<PyHomographyEstimator>()?;
    m.add_class::<PyFundamentalEstimator>()?;
    m.add_class::<PyEssentialEstimator>()?;
    m.add_class::<PyAbsolutePoseEstimator>()?;
    m.add_class::<PyRigidTransformEstimator>()?;
    m.add_class::<PyLineEstimatorNative>()?;
    m.add_class::<PyScoringAdapter>()?;
    m.add_class::<PyNativeSampler>()?;
    m.add_class::<PySamplerAdapter>()?;
    m.add_class::<PyLocalOptimizerAdapter>()?;
    m.add_class::<PyTerminationAdapter>()?;
    m.add_class::<PyInlierSelectorAdapter>()?;
    m.add_class::<PyMetasacSettings>()?;
    m.add_class::<PyPipeline>()?;
    m.add_class::<PyDataMatrix>()?;

    m.add_function(wrap_pyfunction!(estimate_homography_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_fundamental_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_essential_matrix_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_absolute_pose_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_rigid_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(estimate_line_py, m)?)?;
    m.add_function(wrap_pyfunction!(run_metasac, m)?)?;

    let _ = py;
    Ok(())
}
