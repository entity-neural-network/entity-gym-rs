# Example Snake Game with Bevy

This example shows how to expose a Bevy app as an entity-gym environment, use entity-gym to train a neural network to play snake, and then run the resulting neural network inside a Bevy game.
The snake implementation is based on [https://github.com/marcusbuffett/bevy_snake]().

## Overview

I have tried to keep the code as close to the original as possible.
The majority of the code in `lib.rs` is the same as the original `main.rs` file.
Most of the entity-gym specific code is in `src/ai.rs`.
The `src/python.rs` contains some additional code required to export a Python API. The `train.py`, `train.ron`, and `pyproject.toml` files contain the code required to set up a Python environment and run training.

## Usage

Running the game with random actions:

```shell
cargo run
```

Run with a trained neural network ([download link](https://www.dropbox.com/sh/laja5te8t9uojnw/AADqDndrEOzRgtoVzv8EK8Voa?dl=0)):

```shell
cargo run -- --agent-path bevy_snake1m/latest-step000000999424
```

Training a new agent with [enn-trainer](https://github.com/entity-neural-network/enn-trainer) (requires [`poetry`](https://python-poetry.org/) and only works on Linux, Nvidia GPU recommended):

```shell
poetry install
poetry run pip install setuptools==59.5.0
poetry run pip install torch==1.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
poetry run pip install torch-scatter -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
poetry run pip install .
poetry run python train.py --config=train.ron --checkpoint-dir=checkpoints
```

## How it works

To allow the snake game to be played by the AI, we add a `Player` component that wraps an `AnyAgent` object.

```rust
#[derive(Component)]
pub struct Player(pub AnyAgent);
```

When creating our app, we initialize the agent by loading a neural network from a checkpoint:

```rust
        .insert_resource(Player(AnyAgent::rogue_net(
            "norelattn-1m/latest-step000000999424",
        )))
```

The core part of the integration happens inside the `snake_movement_agent` system defined in [`src/ai.rs`]().

```rust
pub(crate) fn snake_movement_agent(
    mut player: ResMut<Player>,
    mut heads: Query<(&mut SnakeHead, &Position)>,
    mut exit: EventWriter<AppExit>,
    segments_res: Res<SnakeSegments>,
    food: Query<(&Food, &Position)>,
    segment: Query<(&SnakeSegment, &Position)>,
) {
    if let Some((mut head, head_pos)) = heads.iter_mut().next() {
        let obs = Obs::new(segments_res.len() as f32)
            .entities("Food", food.iter().map(|(_, pos)| pos))
            .entities("Head", [head_pos].into_iter())
            .entities("SnakeSegment", segment.iter().map(|(_, pos)| pos));
        let action = player.0.act::<Move>(obs);
        match action {
            Some(Move(dir)) => {
                if dir != head.direction.opposite() {
                    head.direction = dir;
                }
            }
            None => exit.send(AppExit),
        }
    }
}
```

Let's disect what's going on here.
The key line is `let action = player.0.act::<Move>(obs);`, which passes a snapshot the environment to the agent and returns us an `Move` action.
The `Move` action is simply a wrapper around a `Direction` enum:

```rust
struct Move(Direction);
```

Additionally, `Move` implements the `Action` trait:

```rust
impl Action for Move {
    fn from_u64(index: u64) -> Self {
        match index {
            0 => Move(Direction::Up),
            1 => Move(Direction::Down),
            2 => Move(Direction::Left),
            3 => Move(Direction::Right),
            _ => panic!("Invalid direction index"),
        }
    }

    fn to_u64(&self) -> u64 {
        match self.0 {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        }
    }

    fn num_actions() -> u64 {
        4
    }
}
```

This basically just allows `Move` to be converted to/from an integer index and tells the trainig framework about the number of valid actions.
This could just be done automatically by a derive macro.

The input to the agent is an `Obs` struct:

```rust
Obs::new(segments_res.len() as f32)
    .entities("Food", food.iter().map(|(_, pos)| pos))
    .entities("Head", [head_pos].into_iter())
    .entities("SnakeSegment", segment.iter().map(|(_, pos)| pos));
```

The argument taken by `Obs::new` is the current "score" achieved by the agent, which is used as the reward signal during training.
We want the agent to create as long of a snake as possible, so we use the number of segments as the score.

Additionally, we supply the `Obs` with several lists of entities that we want the agent to see.
The entities must implement the `Featurizable` trait that allows them to be converted into a flat list of floating point numbers (again, this could be done automatically by a derive macro):

```rust
impl Featurizable for Position {
    fn num_feats() -> usize {
        2
    }

    fn feature_names() -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }

    fn featurize(&self) -> Vec<f32> {
        vec![self.x as f32, self.y as f32]
    }
}
```

One final detail is that the `act` method may return `None`.
This happens if we are running in training mode and the framework decides close the environment.
In this case, we should terminate the application.

## Training

To support training, we define a `run_headless` method at the end of [`src/lib.rs`]() that creates a version of our app that runs as fast as possible and doesn't render any graphics.
We also need some additional boilerplate that exports a Python interface, defined in [`src/python.rs`](). There are three interesting bits:

1. We define a `Config` struct that allows us to pass configuration values to our game from Python (not actually used for anything here).

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

2. We define the interface for our environment (this must match what the way we construct the `Obs` struct):

```rust
fn env(_config: Config) -> (TrainAgentEnv, TrainAgent) {
    TrainEnvBuilder::default()
        .entity::<Position>("Head")
        .entity::<Position>("SnakeSegment")
        .entity::<Position>("Food")
        .build()
}
```

3. We define a closure that can be used to create and run a new app instance:

```rust
Arc::new(move |seed| {
    let (env, agent) = env(config.clone());
    thread::spawn(move || {
        super::run_headless(AnyAgent::train(agent), seed);
    });
    env
})
```

## Other notes

- The original snake implementation used an event timer to spawn a food every second.
  This doesn't really work when we want to run the game in headless mode as fast as possible, so I added counter resource to keep track of logical time and spawn a food on every 7th tick instead.
- When training, we crate and run many parallel game instances.
  If all of these have the same starting state, they can end up generating identical or highly correlated trajectories which degrades the training.
  For this reason, the `run_headless` method to receives a `seed` (which will be different for each App instance) and use it to randomize the starting.