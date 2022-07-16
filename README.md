# EntityGym for Rust

[EntityGym](https://github.com/entity-neural-network/entity-gym) is a Python library that defines a novel entity-based abstraction for reinforcement learning environments, which enables highly ergonomic and efficient training of deep reinforcement learning agents.
This crate provides bindings that allows Rust programs to implement the entity-gym API.

## Background

Most RL libraries (including entity-gym) define an API that looks something like this:

```python
class Environment:
    # Reset the environment the initial state and return the first observation.
    def reset(self) -> Obs:
        ...
    # Perform an action and return the reward and observation for the next step.
    def step(self, action: Action) -> (Obs, f32):
        ...
    # Defines the shape of observations.
    def obs_space(&self) -> ObsSpace:
        ...
    # Defines the kinds of actions that can be performed.
    def act_space(&self) -> ActSpace:
        ...
```

RL training frameworks then run a loop that is essentially:

```python
def train(steps: int):
    envs = [Environment() for _ in range(num_envs)]
    obs = [env.reset() for env in envs]

    for step in 0..steps:
        actions = deep_neural_network(obs)
        # Instead of iterating over environments, most real-world RL
        # frameworks will use a "vectorized" environment here which
        # allows for more efficient/parallelized stepping of multiple environments.
        obs, rewards = unzip(env.step(action) for env, action in zip(envs, actions))
```

## `low_level`

The [`low_level` module](src/low_level/) defines an interface that closely mirrors the abstractions in entity-gym.

The `Environment` ([low-level/env.rs](src/low_level/env.rs)) trait defines the main abstraction that needs to be implemented by the user.
It is similar to the `Environment` trait in the [`entity-gym`](), but somewhat lower-level to allow for maximum efficiency (I found that even allocating hashmaps on each step is too expensive).
As a result of these optimizations (and some other issues), the `Environment` interface currently very cumbersome and difficult to use correctly, though I think this could mostly be fixed without comproming efficiency.

The [`VecEnv`](src/low_level/vec_env.rs) is an efficient multi-threaded executor which exposes a vectorized `Environment` interface that allows multiple environments to be stepped in a single call.
Since a single environment step can be a very cheap operation (cheaper than thread synchronization/context switching overhead), `VecEnv` is aggressively optimized to still benefit from parallelization.
After a lot of experimentation and profiling, I arrived at the following strategy:
- `VecEnv` spawns a fixed number of worker threads, each of which owns several `Environment` instances.
- At entry, the `VecEnv::act` (step) method sets an atomic counter to the number of worker threads.
- The main thread iterates over a list of channels (one for each worker thread) and sends an Arc of the action batch to each worker.
- The main thread suspends itself using a `crossbeam::sync::Parker`.
- When a worker thread receives the next batch of actions, it synchronously steps all the environments it owns.
- All worker threads and the main thread share a single statically allocated vector that holds one atomic pointer for each environment. This is where we store the observation returned by the environment instances.
- Once a worker is done, it decrements the atomic counter, unparks the main thread if the counter has reached zero, and awaits the next batch of actions.
- The main thread collects all observations and returns them to the caller.

The `PyVecEnv` is a thin wrapper around `VecEnv` that performs some additional conversions to and from Python types.

## `agent`

Various lower-level issues with the `Environment` trait aside, its high-level interface also doesn't seem to mesh well with the typical structure of Bevy apps.
Particularly, it assumes that the environment is stepped externally and yields control back to the main thread after each step.
I didn't find a good way to directly implement the `Environment` interface for a Bevy app, though someone more familiar with how Bevy works might know a solution.
What seemed much more natural to me was to create a [`Agent`](src/agent/agent.rs) interface that inverts the control flow, allowing the environment to request actions rather than the training framework requesting observations.

```rust
pub trait Agent {
    fn act<A: Action>(&mut self, obs: Obs) -> Option<A>;
    fn game_over(&mut self) {}
}
```

The `Agent` trait has two primary implementations:
- The [`RogueNetAgent`](src/agent/rogue_net_agent.rs) wraps a neural network loaded from an enn-trainer checkpoint that can directly generate actions.
- The [`TrainAgent`](src/agent/env.rs) is a bidirectional channel that can interface with an external training process. When act is called on this agent, it sends the observation on a channel and the blocks on another channel to await the next action. Each `TrainAgent` has a corresponding `TrainAgentEnv` counterpart which implements the `Environment` interface sending actions and receiving observations.