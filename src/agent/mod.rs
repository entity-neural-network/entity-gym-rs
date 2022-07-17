mod action;
mod any_agent;
mod env;
mod featurizable;
mod obs;
mod random_agent;
mod rogue_net_agent;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use action::Action;
pub use any_agent::AnyAgent;
use crossbeam_channel::Receiver;
pub use env::{TrainAgent, TrainAgentEnv, TrainEnvBuilder};
pub use featurizable::Featurizable;
pub use obs::Obs;
pub use random_agent::RandomAgent;
pub use rogue_net_agent::RogueNetAgent;

pub trait Agent {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A>;
    #[must_use]
    fn act_async<A: Action>(&mut self, obs: &Obs) -> ActionReceiver<A>;
    fn game_over(&mut self, _score: f32) {}
}

pub struct ActionReceiver<A> {
    inner: InnerActionReceiver<A>,
}

enum InnerActionReceiver<A> {
    Receiver {
        receiver: Receiver<u64>,
        observations_remaining: Arc<AtomicUsize>,
        agent_count: usize,
    },
    Value(A),
}

impl<A: Action> ActionReceiver<A> {
    pub fn rcv(self) -> Option<A> {
        match self.inner {
            InnerActionReceiver::Receiver {
                receiver,
                observations_remaining,
                agent_count,
            } => {
                let remaining = observations_remaining.load(Ordering::SeqCst);
                if remaining != 0 && remaining != agent_count {
                    panic!("TrainAgent::act called before all agents have received observations. This is not allowed. If you have multiple agents, call the `act_async` on every agent before awaiting any actions.");
                }
                let act = receiver.recv();
                observations_remaining.store(agent_count, Ordering::SeqCst);
                act.ok().map(A::from_u64)
            }
            InnerActionReceiver::Value(action) => Some(action),
        }
    }

    pub fn value(val: A) -> ActionReceiver<A> {
        ActionReceiver {
            inner: InnerActionReceiver::Value(val),
        }
    }
}
