# EntityGym for Rust

[![Crates.io](https://img.shields.io/crates/v/entity-gym-rs.svg)](https://crates.io/crates/entity-gym-rs)
[![MIT/Apache 2.0](https://img.shields.io/badge/license-MIT%2FApache-blue.svg)](./LICENSE)
[![Crates.io](https://img.shields.io/crates/d/entity-gym-rs.svg)](https://crates.io/crates/entity-gym-rs)
[![Actions Status](https://github.com/entity-neural-network/entity-gym-rs/workflows/Test/badge.svg)](https://github.com/entity-neural-network/entity-gym-rs/actions)
[![Discord](https://img.shields.io/discord/913497968701747270?style=flat-square)](https://discord.gg/SjVqhSW4Qf)


[EntityGym](https://github.com/entity-neural-network/entity-gym) is a Python library that defines a novel entity-based abstraction for reinforcement learning environments which enables highly ergonomic and efficient training of deep reinforcement learning agents.
This crate provides bindings that allows Rust programs to be used as EntityGym training environments, and to load and run neural networks agents trained with [Entity Neural Network Trainer](https://github.com/entity-neural-network/enn-trainer) inside Rust.

## Overview

The entity-gym-rs crate provides a high-level API that allows neural network agents to interact directly with Rust data structures.

```rust
use entity_gym_rs::agent::{Agent, AgentOps, Obs, Action, Featurizable};

// The `Action` trait can be automatically derived for any enum with only unit variants.
#[derive(Action, Debug)]
enum Move { Up, Down, Left, Right }

// The `Featurizable` trait can be automatically derived for any struct that contains
// only primitive number types, booleans, or other `Featurizable` types.
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
    // To load an neural network agent from an enn-trainer checkpoint, you would use the `load` method instead.
    // let mut agent = Agent::load("agent");

    // An observation can be constructed from any number of `Featurizable` objects.
    let obs = Obs::new(0.0)
        .entities([Player { x: 0, y: 0 }])
        .entities([
            Cake { x: 4, y: 0, size: 4 },
            Cake { x: 10, y: 42, size: 12 },
        ]);
    
    // The agent `act` method takes an observation and returns an action of the specified type.
    let action = agent.act::<Move>(obs);
    println!("{:?}", action);
}
```

## Docs

- [bevy_snake](examples/bevy_snake): Example of how to use entity-gym-rs in a Bevy game.
- [bevy_multisnake](examples/bevy_snake): Example of more advanced Bevy integration and adversarial training with multiple agents.
- [EntityGym Rust API Docs](https://docs.rs/entity-gym-rs/0.1.0/entity_gym_rs/): Rust API reference.
