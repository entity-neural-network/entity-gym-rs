use std::sync::Arc;
use std::thread;

use crate::ai::{self};
use crate::Direction;
use entity_gym_rs::agent::{TrainAgent, TrainAgentEnv, TrainEnvBuilder};
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

pub fn env(_config: Config) -> (TrainAgentEnv, Vec<TrainAgent>) {
    TrainEnvBuilder::default()
        .entity::<ai::Head>()
        .entity::<ai::SnakeSegment>()
        .entity::<ai::Food>()
        .action::<Direction>()
        .build_multiagent(2)
}

#[pyfunction]
fn create_env(config: Config, num_envs: usize, threads: usize, first_env_index: u64) -> PyVecEnv {
    PyVecEnv {
        env: VecEnv::new(
            Arc::new(move |seed| {
                let (env, mut agents) = env(config.clone());
                let a1 = Box::new(agents.pop().unwrap());
                let a0 = Box::new(agents.pop().unwrap());
                thread::spawn(move || {
                    super::run_headless([a0, a1], seed);
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
fn bevy_multisnake(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_class::<Config>()?;
    Ok(())
}
