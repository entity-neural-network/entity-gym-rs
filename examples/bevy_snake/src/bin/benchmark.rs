#[cfg(feature = "python")]
use bevy_snake_enn::python::Config;

#[cfg(feature = "python")]
fn main() {
    use bevy_snake_enn::ai::{Food, Head, SnakeSegment};
    use bevy_snake_enn::{run_headless, Direction};
    use entity_gym_rs::agent::TrainEnvBuilder;
    use ragged_buffer::ragged_buffer::RaggedBuffer;
    //use std::hint::black_box;
    use std::time::Instant;

    let start_time = Instant::now();
    let config = Config;
    let mut env = TrainEnvBuilder::default()
        .entity::<Head>()
        .entity::<SnakeSegment>()
        .entity::<Food>()
        .action::<Direction>()
        .build(config, run_headless, 128, 4, 0)
        .env;
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
