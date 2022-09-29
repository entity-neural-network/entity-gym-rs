use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::{ActionReceiver, Agent, Obs};

/// Agent that samples all actions uniformly at random.
pub struct RandomAgent {
    rng: SmallRng,
}

impl Agent for RandomAgent {
    fn act_dyn(&mut self, _action: &str, num_actions: u64, obs: &Obs) -> Option<Vec<u64>> {
        let actions = (0..obs.num_actors())
            .map(|_| self.rng.gen_range(0..num_actions))
            .collect();
        Some(actions)
    }

    fn act_async_dyn(&mut self, _action: &str, num_actions: u64, obs: &Obs) -> ActionReceiver<u64> {
        let actions = (0..obs.num_actors())
            .map(|_| self.rng.gen_range(0..num_actions))
            .collect();
        ActionReceiver::value(actions)
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
