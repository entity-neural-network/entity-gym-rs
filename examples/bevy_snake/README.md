# Example Snake Game with Bevy

This example shows how to expose a Bevy app as an entity-gym environment, use entity-gym to train a neural network to play snake, and then run the resulting neural network inside a Bevy game.
The snake implementation is based on  [Marcus Buffett's snake clone](https://mbuffett.com/posts/bevy-snake-tutorial/).

## Overview

The majority of the code is mostly unchanged from the [original implementation](https://github.com/marcusbuffett/bevy_snake/tree/c0344a40d28eb321493ee950e64ecc5bca6cc5a4):
- The `main.rs` file has been renamed to `lib.rs`, with the new entry point moved to `bin/main.rs`.
- The new AI controller lives in [`src/ai.rs`](src/ai.rs).
- The additional code required for training is in [`src/python.rs`](src/python.rs), which defines a PyO3 Python API. The [`train.py`](train.py) is simple script to run training, [`train.ron`](train.ron) defines some hyperparameters, and [`pyproject.toml`](pyproject.toml)/[`poetry.lock`](poetry.lock) define required Python dependencies with the [Poetry](https://python-poetry.org/) package manager.

## Usage

Running the game with random actions:

```shell
cargo run
```

Run with a trained neural network ([download link](https://www.dropbox.com/sh/laja5te8t9uojnw/AADqDndrEOzRgtoVzv8EK8Voa?dl=0)):

```shell
cargo run -- --agent-path bevy_snake1m/latest-step000000999424
```

Training a new agent with [enn-trainer](https://github.com/entity-neural-network/enn-trainer) (requires [poetry](https://python-poetry.org/) and only works on Linux. Nvidia GPU with working CUDA installation recommended.):

```shell
poetry install
poetry run pip install setuptools==59.5.0
poetry run pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
poetry run maturin develop --release --features=python
poetry run python train.py --config=train.ron --checkpoint-dir=checkpoints
```

## How it works

This guide lists all steps required to have an AI play the snake game rather than a human.

The first step is to add a new resource to the app which encapsulates the AI player.

The resource is defined in [`src/ai.rs`](src/ai.rs#L32):

```rust
pub struct Player(pub Box<dyn Agent>);
```

The [`Agent`](https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/trait.Agent.html) trait defines the interface for different AI implementations.


To allow the snake game to be played by the AI, we add a `Player` resource that wraps an `AnyAgent` object.
In [`src/lib.rs`](src/lib.rs#L318), we instantiate the `Player` either as a completely randomly acting agent, or as a neural network loaded from a checkpoint:

```rust
        .insert_non_send_resource(match agent_path {
            Some(path) => Player(AnyAgent::rogue_net(&path)),
            None => Player(AnyAgent::random()),
        })
```

The core part of the integration happens inside the [`snake_movement_agent`](src/ai.rs#L7) system.

First, we construct an [`Obs`](https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/struct.Obs.html) structure which defines what parts of the game state are visible to the AI:

```rust
let obs = Obs::new(segments_res.len() as f32)
    .entities(food.iter().map(|(_, p)| Food { x: p.x, y: p.y }))
    .entities([head_pos].iter().map(|p| Head { x: p.x, y: p.y }))
    .entities(segment.iter().map(|(_, p)| SnakeSegment { x: p.x, y: p.y }));
```

The argument to `Obs::new` is the current _score_ of the agent, which is the quantity that will be maximized in the training process.
Since we want the agent to grow the snake as long as possible, we use the number of segments as the score.

The `entities` method allows us to make different entities visible to the AI.
The argument to `entities` is an iterator over [`Featurizable`](https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/trait.Featurizable.html) structs, which is a trait that allows structs to be converted into a representation that can be processed by the neural network.
The [`Featurizable`](https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/trait.Featurizable.html) trait can be derived automatically for any structs that contain only primitive numerals, booleans, and types that implement `Featurizable`:

```rust
#[derive(Featurizable)]
pub struct Head {
    x: i32,
    y: i32,
}

#[derive(Featurizable)]
pub struct SnakeSegment {
    x: i32,
    y: i32,
}

#[derive(Featurizable)]
pub struct Food {
    x: i32,
    y: i32,
}
```

The next part is calling the `act` method on the neural network, which returns an `Option<Direction>` in exchange for the observation we just constructed:

```rust
let action = player.0.act::<Direction>(&obs);
```

The type we use for the action (here, `Direction`) must implement the [`Action`](https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/trait.Action.html) trait.
The `Action` trait can be derived automatically for any `enum` that consists only of unit variants:

```rust
#[derive(PartialEq, Copy, Clone, Debug, Action)]
enum Direction {
    Left,
    Up,
    Right,
    Down,
}
```

Due to a limitation in the current implementation, the `act` method can return `None`, which indicates that we should exit the game.
Otherwise, we simple apply the action to the game the same way we would do with human input:

```rust
match action {
    Some(dir) => {
        if dir != head.direction.opposite() {
            head.direction = dir;
        }
    }
    None => exit.send(AppExit),
}
```

## Training

Training neural network agents requires a version of the game that can interface with Python and run many headless game instances in parallel.

### Headless runner

If you look at [`src/lib.rs`](src/lib.rs#L287), you will see that the original `main` method has been split into three functions:

- [`base_app`](src/lib.rs#L287) defines all the systems which we want to run both in the game and in the headless runner.
- [`run`](src/lib.rs#L310) adds all the systems which we want when running the game, such as as creating a window and handling user input.
- [`run_headless`](src/lib.rs#L310) omits the window and user input, uses the `MinimalPlugins` plugin set, sets up a run loop with 0 wait_duration.


Another difference is that the original snake implementation used an event timer to spawn a food every second.
This doesn't work when running without a fixed framerate, so we instead use a [`FoodTimer`](src/lib.rs#L63) resource to keep track of the time since the last food was spawned and [spawn food on every 7th tick](src/lib.rs#L265).

### Python API

We use [PyO3](https://pyo3.rs) to export the game as a Python module.
There is currently [an issue](https://github.com/PyO3/pyo3/issues/1708) that causes long compile times when using PyO3 as a dependency.
For this reason, we gate all the Python specific code and the PyO3 dependecy behind a "python" feature flag.

```toml
[dependencies]
pyo3 = { version = "0.15", features = ["extension-module"], optional = true }

[features]
python = ["pyo3", "entity-gym-rs/python"]
```

Defining the Python API requires a certain amount of boilerplate, all of which can be found in [`src/python.rs`](src/python.rs).

1. We define a `Config` struct that allows us to pass in game settings from Python (not actually used for anything here).

```rust
#[derive(Clone)]
#[pyclass]
struct Config;

#[pymethods]
impl Config {
    #[new]
    fn new() -> Self {
        Config
    }
}
```

2. The [`spawn_env`](src/python.rs#L21) function usees the [`TrainEnvBuilder`][TrainEnvBuilder] to obtain a [`TrainAgentEnv`][TrainAgentEnv]/[`TrainAgent`][TrainAgent] pair.
It spawns a thread that an instance of the game in headless mode with the training agent and returns the `TrainAgentEnv` which will be used by the Python code to interact with this game instance.
Any observation and action types that will be used must be registered with the `TrainEnvBuilder`.

```rust
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
```

3. The [`create_env`](src/python.rs#L36) function is what we will actually call from Python to create an array of game instances.
It's largely boilerplate, the only part that's specific to the game is the `Config` type and the call to `spawn_env`.

```rust
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
```

Finally, the `#[pymodule]` attribute constructs the Python module and registers the `Config` type and the `create_env` function.

```rust
#[pymodule]
fn bevy_snake_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_class::<Config>()?;
    Ok(())
}
```

[TrainEnvBuilder]: https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/struct.TrainEnvBuilder.html
[TrainAgentEnv]: https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/struct.TrainAgentEnv.html
[TrainAgent]: https://docs.rs/entity-gym-rs/0.1.3/entity_gym_rs/agent/struct.TrainAgent.html