use std::sync::Arc;
use std::thread;

use crate::ai::{self, Move};
use entity_gym_rs::agent::{AnyAgent, TrainAgent, TrainAgentEnv, TrainEnvBuilder};
use entity_gym_rs::low_level::py_vec_env::PyVecEnv;
use entity_gym_rs::low_level::VecEnv;
use pyo3::prelude::*;

#[derive(Clone)]
#[pyclass]
pub struct Config;

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        Config
    }
}

pub fn env(_config: Config) -> (TrainAgentEnv, TrainAgent) {
    TrainEnvBuilder::default()
        .entity::<ai::Head>()
        .entity::<ai::SnakeSegment>()
        .entity::<ai::Food>()
        .action::<Move>()
        .build()
}

#[pyfunction]
fn create_env(config: Config, num_envs: usize, threads: usize, first_env_index: u64) -> PyVecEnv {
    PyVecEnv {
        env: VecEnv::new(
            Arc::new(move |seed| {
                let (env, agent) = env(config.clone());
                thread::spawn(move || {
                    super::run_headless(AnyAgent::train(agent), seed);
                });
                env
            }),
            num_envs,
            threads,
            first_env_index,
        ),
    }
}

#[pymodule]
fn bevy_snake_enn(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_class::<Config>()?;
    Ok(())
}
