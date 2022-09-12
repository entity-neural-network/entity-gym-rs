# Creating an AI to Play a Bevy Snake Game

This example shows how to expose a Bevy app as an [entity-gym](https://github.com/entity-neural-network/entity-gym) environment, use [enn-trainer](https://github.com/entity-neural-network/enn-trainer) to train a neural network to play snake, and then run the resulting neural network as part of a Bevy game.
The snake implementation is lightly modified from [Marcus Buffett's snake clone](https://mbuffett.com/posts/bevy-snake-tutorial/).

## Overview

The majority of the code is mostly unchanged from the [original implementation](https://github.com/marcusbuffett/bevy_snake/tree/c0344a40d28eb321493ee950e64ecc5bca6cc5a4):
- The `main.rs` file has been renamed to `lib.rs`, with the new entry point moved to `bin/main.rs`.
- The new AI controller lives in [`src/ai.rs`](src/ai.rs).
- The additional code required for training is in [`src/python.rs`](src/python.rs), which defines a PyO3 Python API. [`train.py`](train.py) is a simple script that runs training, [`train.ron`](train.ron) defines some hyperparameters, and [`pyproject.toml`](pyproject.toml)/[`poetry.lock`](poetry.lock) define required Python dependencies using the [Poetry](https://python-poetry.org/) package manager.

## Usage

Clone the repo and move to the examples/bevy_snake directory:

```shell
git clone https://github.com/entity-neural-network/entity-gym-rs.git
cd entity-gym-rs/examples/bevy_snake
```

Running the game with random actions:

```shell
cargo run --bin main
```

Run with a trained neural network ([download link](https://www.dropbox.com/s/ctnrkwyz8d3vygk/bevy_snake1m.roguenet?dl=1)):

```shell
cargo run -- --agent-path bevy_snake1m.roguenet
```

Training a new agent with [enn-trainer](https://github.com/entity-neural-network/enn-trainer) (requires [Poetry](https://python-poetry.org/), only tested on Linux, Nvidia GPU recommended):

```shell
poetry install
# Replace "cu113" with "cpu" to train on CPU.
poetry run pip install torch==1.12.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.12.0+cu113.html
poetry run maturin develop --release --features=python
poetry run python train.py --config=train.ron --checkpoint-dir=checkpoints
```

## How it works

This guide will walk you through the steps require to create an AI for the [Bevy snake game](https://mbuffett.com/posts/bevy-snake-tutorial/).

The first step is to add a new resource to the Bevy app which stores the AI player.
The resource is defined in [`src/ai.rs`](src/ai.rs#L32) and holds a `Box<dyn Agent>`:

```rust
pub struct Player(pub Box<dyn Agent>);
```

The [`Agent` trait](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Agent.html) abstracts over different AI implementations provided by entity-gym-rs.

Depending on how the game is configured, we instantiate the `Player` resource in [`src/lib.rs`](src/lib.rs#L319-L322) as either a neural network loaded from a file, or an agent that takes random actions.

```rust
        .insert_non_send_resource(match agent_path {
            Some(path) => Player(agent::load(path)),
            None => Player(agent::random()),
        })
```

The actual integration of the AI player with the game happens inside the [`snake_movement_agent` system](src/ai.rs#L7). This system runs on every tick, obtains actions from the AI, and applies them to the game.
The first step is to construct an [`Obs` structure](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/struct.Obs.html) which collects all the parts of the game state that are visible to the AI:

```rust
let obs = Obs::new(segments_res.len() as f32)
    .entities(food.iter().map(|(_, p)| Food { x: p.x, y: p.y }))
    .entities([head_pos].iter().map(|p| Head { x: p.x, y: p.y }))
    .entities(segment.iter().map(|(_, p)| SnakeSegment { x: p.x, y: p.y }));
```

The argument to `Obs::new` is the current _score_ of the agent, which is the quantity that will be maximized in the training process.
Since we want the agent to grow the snake as long as possible, we use the number of segments as the score.

The `entities` method allows us to make different entities visible to the AI.
The argument to `entities` is an iterator over [`Featurizable`](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Featurizable.html) items, which is a trait that allows structs to be converted into a representation that can be processed by the neural network.
The [`Featurizable`](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Featurizable.html) trait can be derived automatically for enums with unit variants and most fixed-size structs:

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

With the observation constructed, we simply call `act` method on the neural network to obtain an `Option<Direction>`:

```rust
let action = player.0.act::<Direction>(&obs);
```

The type we use for the action (here, `Direction`) must implement the [`Action`](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Action.html) trait.
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
If the result is not `None`, we simple apply the action to the game the same way we would do with human input:

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

- [`base_app`](src/lib.rs#L288) defines all the systems which we want to run both in the game and in the headless runner.
- [`run`](src/lib.rs#L311) adds all the systems which we want when running the game normally, such as as creating a window and handling user input.
- [`run_headless`](src/lib.rs#L335), used during training, omits the window and user input, uses the `MinimalPlugins` plugin set, and sets up a run loop with 0 wait_duration.

Another difference is that the original snake implementation used an event timer to spawn a food every second.
This doesn't work when running without a fixed framerate, so we instead use a [`FoodTimer`](src/lib.rs#L63) resource to keep track of the time since the last food was spawned and [spawn food on every 7th tick](src/lib.rs#L265) instead.

### Python API

We use [PyO3](https://pyo3.rs) to export the game as a Python module.
There is currently [an issue](https://github.com/PyO3/pyo3/issues/1708) that causes long compile times when using PyO3 as a dependency.
For this reason, we gate all the Python specific code and the PyO3 dependecy behind a "python" feature flag.
We also need to build the crate as a `cdylib`.

```toml
[lib]
crate-type = ["cdylib", "rlib"]
name = "bevy_snake_enn"

[dependencies]
pyo3 = { version = "0.15", features = ["extension-module"], optional = true }

[features]
python = ["pyo3", "entity-gym-rs/python"]
```

All the code that is required to define the Python API is in [`src/python.rs`](src/python.rs).
It defines a `Config` struct that allows us to pass in game settings from Python (not actually used for anything in this case).

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

The [`create_env`](src/python.rs#L21) function uses the [`TrainEnvBuilder`][TrainEnvBuilder] to construct a [`PyVecEnv`][PyVecEnv] which runs multiple instances of the game in parallel and will be used directly by the Python training framework in [train.py](train.py).
The `TrainEnvBuilder` requires us to declar the types of all the entities and actions that we want to use in the game using the `entity` and `action` methods.
When we pass the [`run_headless`](src/lib.rs#L310) function to `build`, the `TrainEnvBuilder` will spawn one thread for each environment that calls `run_headless` with a clone of the `Config`, a `TrainAgent` that connects the game to the Python training framework, and a random seed.
The `num_envs`, `threads`, and `first_env_index` parameters are simply forwarded from Python and allow the training framework to control the number of worker threads and game instances.

```rust
#[pyfunction]
fn create_env(config: Config, num_envs: usize, threads: usize, first_env_index: u64) -> PyVecEnv {
    TrainEnvBuilder::default()
        .entity::<ai::Head>()
        .entity::<ai::SnakeSegment>()
        .entity::<ai::Food>()
        .action::<Direction>()
        .build(
            config,
            super::run_headless,
            num_envs,
            threads,
            first_env_index,
        )
}

```

Finally, the `#[pymodule]` macro constructs the Python module and registers the `Config` type and the `create_env` function.

```rust
#[pymodule]
fn bevy_snake_ai(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(create_env, m)?)?;
    m.add_class::<Config>()?;
    Ok(())
}
```

With this we can now run `maturin develop --release --features=python` to build and install the crate as a Python package, which is then imported by the [training script](train.py).

[TrainEnvBuilder]: https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/struct.TrainEnvBuilder.html
[TrainAgentEnv]: https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/struct.TrainAgentEnv.html
[TrainAgent]: https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/struct.TrainAgent.html
[TrainAgent]: https://docs.rs/entity-gym-rs/latest/entity_gym_rs/low_level/struct.PyVecEnv.html
