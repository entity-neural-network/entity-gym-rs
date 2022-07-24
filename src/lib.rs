#![cfg_attr(docsrs, feature(doc_cfg))]
//! # EntityGym for Rust
//!
//! [EntityGym](https://github.com/entity-neural-network/entity-gym) is a Python library that defines a novel entity-based abstraction for reinforcement learning environments which enables highly ergonomic and efficient training of deep reinforcement learning agents.
//! This crate provides bindings that allows Rust programs to implement the entity-gym API and run neural network agents trained with [enn-trainer](https://github.com/entity-neural-network/enn-trainer).

/// High level API for interacting with neural network agents.
pub mod agent;
mod examples;
/// Low-level and highly API that mirrors the entity-gym Python API. Not intended for direct use.
pub mod low_level;

#[cfg(feature = "python")]
mod python {
    use std::sync::Arc;

    use crate::examples::multisnake::MultiSnake;

    use self::py_vec_env::PyVecEnv;

    pub use super::low_level::*;
    use pyo3::prelude::*;

    #[pyfunction(
        board_size = "10",
        first_env_index = "0",
        num_snakes = "1",
        max_snake_length = "10",
        max_steps = "100"
    )]
    fn multisnake(
        num_envs: usize,
        threads: usize,
        board_size: usize,
        first_env_index: u64,
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
                first_env_index,
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
