#[cfg(feature = "python")]
use bevy_snake_enn::python::{env, Config};

#[cfg(feature = "python")]
fn main() {
    use bevy_snake_enn::run_headless;
    use entity_gym_rs::low_level::VecEnv;
    use ragged_buffer::ragged_buffer::RaggedBuffer;
    //use std::hint::black_box;
    use std::sync::Arc;
    use std::thread;
    use std::time::Instant;

    let start_time = Instant::now();
    let config = Config;
    let mut env = VecEnv::new(
        Arc::new(move |seed| {
            let (env, agent) = env(config.clone());
            thread::spawn(move || {
                run_headless(Box::new(agent), seed);
            });
            env
        }),
        128,
        4,
        0,
    );
    let steps = 2000;
    env.reset();
    for i in 0..steps {
        let _obs = env.act(vec![Some(RaggedBuffer::<i64> {
            data: (0..128).map(|j| j * i * 991 % 4).collect(),
            subarrays: (0..128).map(|i| i..i + 1).collect(),
            features: 1,
            items: 128,
        })]);
        //black_box(obs);
    }
    let throughput =
        steps as f64 * env.num_envs as f64 / (start_time.elapsed().as_secs() as f64) / 1000.0;
    println!("{} K samples/s", throughput);
}
 
#[cfg(not(feature = "python"))]
fn main() {
    println!("Compile with --features=python");
}