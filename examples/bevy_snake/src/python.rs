use std::sync::Arc;
use std::thread;

use crate::ai::{self};
use crate::Direction;
use entity_gym_rs::agent::{TrainAgentEnv, TrainEnvBuilder};
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

pub fn spawn_env(_config: Config, seed: u64) -> TrainAgentEnv {
    let (env, agent) = TrainEnvBuilder::default()
        .entity::<ai::Head>()
        .entity::<ai::SnakeSegment>()
        .entity::<ai::Food>()
        .action::<Direction>()
        .build();
    thread::spawn(move || {
        super::run_headless(Box::new(agent), seed);
    });
    env
}

#[pyfunction]
fn create_env(config: Config, num_envs: usize, threads: usize, first_env_index: u64) -> PyVecEnv {
    PyVecEnv {
        env: VecEnv::new(
            Arc::new(move |seed| spawn_env(config.clone(), seed)),
            num_envs,
            threads,
            first_env_index,
        ),
    }
}

#[pymodule]
fn bevy_snake_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_class::<Config>()?;
    Ok(())
}
