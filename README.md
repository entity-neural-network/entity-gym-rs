# EntityGym for Rust

[![Crates.io](https://img.shields.io/crates/v/entity-gym-rs.svg?style=flat-square)](https://crates.io/crates/entity-gym-rs)
[![PyPI](https://img.shields.io/pypi/v/entity-gym-rs.svg?style=flat-square)](https://pypi.org/project/entity-gym-rs/)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg?style=flat-square)](./LICENSE)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)
[![Docs](https://docs.rs/entity-gym-rs/badge.svg?style=flat-square)](https://docs.rs/entity-gym-rs)
[![Actions Status](https://github.com/entity-neural-network/entity-gym-rs/workflows/Test/badge.svg)](https://github.com/entity-neural-network/entity-gym-rs/actions)

[EntityGym](https://github.com/entity-neural-network/entity-gym) is a Python library that defines a novel entity-based abstraction for reinforcement learning environments which enables highly ergonomic and efficient training of deep reinforcement learning agents.
This crate provides bindings that allows Rust programs to be used as EntityGym training environments, and to load and run neural networks agents trained with [Entity Neural Network Trainer](https://github.com/entity-neural-network/enn-trainer) natively in pure Rust applications.

## Overview

The core abstraction in entity-gym-rs is the [`Agent` trait](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.AgentOps.html).
It defines a high-level API for neural network agents which allows them to directly interact with Rust data structures.
To use any of the `Agent` implementations provided by entity-gym-rs, you just need to derive the `Action` and `Featurizable` traits, which define what information the agent can observe and what actions it can take:

- The [`Action` trait](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Action.html) allows a Rust type to be returned as an action by an `Agent`. This trait can be derived automatically for enums with only unit variants.
- The [`Featurizable` trait](https://docs.rs/entity-gym-rs/latest/entity_gym_rs/agent/trait.Featurizable.html) converts objects into a format that can be processed by neural networks. It can be derived for most fixed-size `struct`s and enums with unit variants. `Agent`s can observe collections containing any number of `Featurizable` objects.

## Example

Basic example that demonstrates how to construct an observation and sample a random action from an `Agent`:

```rust
use entity_gym_rs::agent::{Agent, AgentOps, Obs, Action, Featurizable};

#[derive(Action, Debug)]
enum Move { Up, Down, Left, Right }

#[derive(Featurizable)]
struct Player { x: i32, y: i32 }

#[derive(Featurizable)]
struct Cake {
    x: i32,
    y: i32,
    size: u32,
}

fn main() {
    // Creates an agent that acts completely randomly.
    let mut agent = Agent::random();
    // Alternatively, load a trained neural network agent from a checkpoint.
    // let mut agent = Agent::load("agent");

    // Construct an observation with one `Player` entity and two `Cake entities.
    let obs = Obs::new(0.0)
        .entities([Player { x: 0, y: 0 }])
        .entities([
            Cake { x: 4, y: 0, size: 4 },
            Cake { x: 10, y: 42, size: 12 },
        ]);
    
    // To obtain an action from an agent, we simple call the `act` method
    with the observation we constructed.
    let action = agent.act::<Move>(obs);
    println!("{:?}", action);
}
```

For a more complete example that includes training a neural network to play Snake, see [examples/bevy_snake](examples/bevy_snake).  

## Docs

- [bevy_snake](examples/bevy_snake): Example of how to use entity-gym-rs in a Bevy game.
- [bevy-snake-ai](https://github.com/cswinter/bevy-snake-ai): More complex Bevy application with adversarial training of multiple agents to create AI opponents.
- [EntityGym Rust API Docs](https://docs.rs/entity-gym-rs/0.1.0/entity_gym_rs/): Rust API reference.
- If you have any questions, you can also get help on [our discord server](https://discord.gg/SjVqhSW4Qf)
