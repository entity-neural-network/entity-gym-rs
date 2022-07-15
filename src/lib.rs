pub mod agent;
pub mod low_level;

#[cfg(feature = "python")]
mod python {
    pub use super::low_level::*;
    use pyo3::prelude::*;

    #[pymodule]
    fn entity_env_rs(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<py_vec_env::VecObs>()?;
        m.add_class::<py_vec_env::PyVecEnv>()?;
        Ok(())
    }
}
