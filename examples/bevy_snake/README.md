# Example Snake Game with Bevy

This example shows how to expose a Bevy app as an entity-gym environment, use entity-gym to train a neural network to play snake, and then run the resulting neural network inside a Bevy game.
The snake implementation is based on [https://github.com/marcusbuffett/bevy_snake]().

## Interface/Changes
- agent and action and feature stuff
- headless
- pyo3 boilerplate
- logical time

## Entity gym training

## Using trained agent

## Design considerations

- VecEnv vs Agent abstractions
- Current Agent lifetime limitations
- Featurize trait
- Register entity types with string or separate structs
- Problem: components must be sync/send (which channel isn't), therefore need mutex on agent
- multiple inits, env termination?
- agent interface: connection loss/exit signal
- how to expose config to python (register struct with entity-gym?)
- Action/Obs types as generics vs objects