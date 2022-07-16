pub mod agent;
mod examples;
pub mod low_level;

//#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use crate::examples::multisnake::MultiSnake;

    use self::py_vec_env::PyVecEnv;

    pub use super::low_level::*;
    use pyo3::prelude::*;

    #[pyfunction]
    fn multisnake(
        num_envs: usize,
        threads: usize,
        initial_seed: u64,
        board_size: usize,
        num_snakes: usize,
        max_snake_length: usize,
        max_steps: usize,
    ) -> PyVecEnv {
        PyVecEnv {
            env: VecEnv::new(
                Arc::new(move |i| {
                    MultiSnake::new(board_size, num_snakes, max_snake_length, max_steps, i)
                }),
                num_envs,
                threads,
                initial_seed,
            ),
        }
    }

    #[pymodule]
    fn entity_gym_rs(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_class::<py_vec_env::VecObs>()?;
        m.add_class::<py_vec_env::PyVecEnv>()?;
        m.add_function(wrap_pyfunction!(multisnake, m)?)?;
        Ok(())
    }
}
