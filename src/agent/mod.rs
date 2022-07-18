mod action;
mod train_env;
mod featurizable;
mod obs;
mod random_agent;
mod rogue_net_agent;

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use action::Action;
use crossbeam_channel::Receiver;
pub use entity_gym_derive::*;
pub use train_env::{TrainAgent, TrainAgentEnv, TrainEnvBuilder};
pub use featurizable::Featurizable;
pub use obs::Obs;
pub use random_agent::RandomAgent;
pub use rogue_net_agent::RogueNetAgent;

pub trait Agent {
    fn act_raw(&mut self, action: &str, num_actions: u64, obs: &Obs) -> Option<u64>;
    #[must_use]
    fn act_async_raw(&mut self, action: &str, num_actions: u64, obs: &Obs) -> ActionReceiver<u64>;
    fn game_over(&mut self, _score: f32) {}
}

pub trait AgentOps {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A>;
    #[must_use]
    fn act_async<A: Action>(&mut self, obs: &Obs) -> ActionReceiver<A>;
}

impl AgentOps for dyn Agent {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A> {
        self.act_raw(A::name(), A::num_actions(), obs)
            .map(A::from_u64)
    }

    #[must_use]
    fn act_async<A: Action>(&mut self, obs: &Obs) -> ActionReceiver<A> {
        let receiver = self.act_async_raw(A::name(), A::num_actions(), obs);
        unsafe { std::mem::transmute::<ActionReceiver<u64>, ActionReceiver<A>>(receiver) }
    }
}

pub struct ActionReceiver<A> {
    inner: InnerActionReceiver<A>,
}

enum InnerActionReceiver<A> {
    Receiver {
        receiver: Receiver<u64>,
        observations_remaining: Arc<AtomicUsize>,
        agent_count: usize,
        phantom: std::marker::PhantomData<A>,
    },
    Value(u64),
}

impl<A> ActionReceiver<A> {
    pub fn rcv_raw(self) -> Option<u64> {
        match self.inner {
            InnerActionReceiver::Receiver {
                receiver,
                observations_remaining,
                agent_count,
                ..
            } => {
                let remaining = observations_remaining.load(Ordering::SeqCst);
                if remaining != 0 && (remaining != agent_count || agent_count == 1) {
                    panic!("TrainAgent::act called before all agents have received observations. This is not allowed. If you have multiple agents, call the `act_async` on every agent before awaiting any actions.");
                }
                let act = receiver.recv();
                observations_remaining.store(agent_count, Ordering::SeqCst);
                act.ok()
            }
            InnerActionReceiver::Value(value) => Some(value),
        }
    }

    pub fn rcv(self) -> Option<A>
    where
        A: Action,
    {
        self.rcv_raw().map(A::from_u64)
    }

    pub(crate) fn value(val: u64) -> ActionReceiver<A> {
        ActionReceiver {
            inner: InnerActionReceiver::Value(val),
        }
    }
}
