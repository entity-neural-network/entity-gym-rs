mod action;
mod any_agent;
mod env;
mod featurizable;
mod obs;
mod random_agent;
mod rogue_net_agent;

pub use action::Action;
pub use any_agent::AnyAgent;
pub use env::{TrainAgent, TrainAgentEnv, TrainEnvBuilder};
pub use featurizable::Featurizable;
pub use obs::Obs;
pub use random_agent::RandomAgent;
pub use rogue_net_agent::RogueNetAgent;

pub trait Agent {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A>;

    fn game_over(&mut self) {}
}
