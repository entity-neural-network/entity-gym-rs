use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::{ActionReceiver, Agent, Obs};

/// Agent that samples all actions uniformly at random.
pub struct RandomAgent {
    rng: SmallRng,
}

impl Agent for RandomAgent {
    fn act_dyn(&mut self, _action: &str, num_actions: u64, _: &Obs) -> Option<u64> {
        Some(self.rng.gen_range(0..num_actions))
    }

    fn act_async_dyn(&mut self, _action: &str, num_actions: u64, _: &Obs) -> ActionReceiver<u64> {
        ActionReceiver::value(self.rng.gen_range(0..num_actions))
    }

    fn game_over(&mut self, _: &Obs) {}
}

impl RandomAgent {
    /// Creates a new `RandomAgent` with a random seed.
    pub fn from_seed(seed: u64) -> Self {
        RandomAgent {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        RandomAgent::from_seed(0)
    }
}
