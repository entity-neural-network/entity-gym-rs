mod action;
mod featurizable;
mod obs;
mod random;
mod rogue_net;
#[cfg(feature = "bevy")]
mod rogue_net_asset;
#[cfg(feature = "python")]
mod training;

use std::io::Read;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

pub use self::rogue_net::RogueNetAgent;
pub use action::Action;
use crossbeam_channel::Receiver;
pub use entity_gym_derive::*;
pub use featurizable::Featurizable;
pub use obs::Obs;
pub use random::RandomAgent;
#[cfg(feature = "bevy")]
pub use rogue_net_asset::{RogueNetAsset, RogueNetAssetLoader};
#[cfg(feature = "python")]
pub use training::{TrainAgent, TrainAgentEnv, TrainEnvBuilder};

/// Agents are given observations and return actions.
///
/// You don't generally need to implement the [`Agent`] trait yourself.
/// There are three main ways of obtaining an [`Agent`]:
/// 1. [`random()`] creates an agent that chooses actions uniformly at random.
/// 2. [`load`] and [`load_archive`] loads a trained neural network agent from an [enn-trainer](https://github.com/entity-neural-network/enn-trainer) checkpoint directory or an archive of a checkpoint directory.
/// 3. [`TrainEnvBuilder`] can be used to obtain a [`TrainAgent`]/[`TrainAgentEnv`] pair which can be used to train a neural network agent.
///
/// Every [`Agent`] also implements the [`AgentOps`] trait which provides more ergonomic typed versions of the [`Agent::act_dyn`] and [`Agent::act_async_dyn`] methods.
pub trait Agent {
    /// Returns an action for the given observation.
    fn act_dyn(&mut self, action: &str, num_actions: u64, obs: &Obs) -> Option<u64>;

    /// Returns receiver that can be blocked on to receive an action for the given observation.
    #[must_use]
    fn act_async_dyn(&mut self, action: &str, num_actions: u64, obs: &Obs) -> ActionReceiver<u64>;

    /// Indicates that the agent has reached the end of the training episode.
    fn game_over(&mut self, obs: &Obs);
}

/// Augments the [`Agent`] trait with more ergonomic typed versions of the [`Agent::act_dyn`] and [`Agent::act_async_dyn`] methods.
pub trait AgentOps {
    /// Returns an action for the given observation.
    fn act<'a, A: Action<'a>>(&mut self, obs: &Obs) -> Option<A>;

    /// Returns receiver that can be blocked on to receive an action for the given observation.
    #[must_use]
    fn act_async<'a, A: Action<'a>>(&mut self, obs: &Obs) -> ActionReceiver<A>;
}

impl<T: Agent> AgentOps for T {
    fn act<'a, A: Action<'a>>(&mut self, obs: &Obs) -> Option<A> {
        self.act_dyn(A::name(), A::num_actions(), obs)
            .map(A::from_u64)
    }

    #[must_use]
    fn act_async<'a, A: Action<'a>>(&mut self, obs: &Obs) -> ActionReceiver<A> {
        let receiver = self.act_async_dyn(A::name(), A::num_actions(), obs);
        unsafe { std::mem::transmute::<ActionReceiver<u64>, ActionReceiver<A>>(receiver) }
    }
}

impl AgentOps for dyn Agent {
    fn act<'a, A: Action<'a>>(&mut self, obs: &Obs) -> Option<A> {
        self.act_dyn(A::name(), A::num_actions(), obs)
            .map(A::from_u64)
    }

    #[must_use]
    fn act_async<'a, A: Action<'a>>(&mut self, obs: &Obs) -> ActionReceiver<A> {
        let receiver = self.act_async_dyn(A::name(), A::num_actions(), obs);
        unsafe { std::mem::transmute::<ActionReceiver<u64>, ActionReceiver<A>>(receiver) }
    }
}

/// A channel for receiving an agent action returned by [`AgentOps::act_async`] or [`Agent::act_async_dyn`].
pub struct ActionReceiver<A> {
    inner: InnerActionReceiver<A>,
}

enum InnerActionReceiver<A> {
    // Variant is only constructed when cfg(feature = "python").
    #[allow(dead_code)]
    Receiver {
        receiver: Receiver<u64>,
        observations_remaining: Arc<AtomicUsize>,
        agent_count: usize,
        phantom: std::marker::PhantomData<A>,
    },
    Value(u64),
}

impl<A> ActionReceiver<A> {
    /// Blocks on the receiver until an action is received.
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

    /// Blocks on the receiver until an action is received.
    pub fn rcv<'a>(self) -> Option<A>
    where
        A: Action<'a>,
    {
        self.rcv_raw().map(A::from_u64)
    }

    /// Blocks on the receiver until an action is received.
    pub(crate) fn value(val: u64) -> ActionReceiver<A> {
        ActionReceiver {
            inner: InnerActionReceiver::Value(val),
        }
    }
}

/// Returns a boxed [`RandomAgent`].
pub fn random() -> Box<dyn Agent> {
    Box::new(RandomAgent::default())
}

/// Returns a boxed [`RandomAgent`] with the given seed.
pub fn random_seeded(seed: u64) -> Box<dyn Agent> {
    Box::new(RandomAgent::from_seed(seed))
}

/// Loads an agent from a checkpoint directory.
pub fn load<P: AsRef<Path>>(path: P) -> Box<dyn Agent> {
    Box::new(RogueNetAgent::load(path).unwrap())
}

/// Loads an agent from an archive of a checkpoint directory.
pub fn load_archive<R: Read>(reader: R) -> Result<Box<dyn Agent>, std::io::Error> {
    Ok(Box::new(RogueNetAgent::load_archive(reader)?))
}
