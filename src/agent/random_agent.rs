use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::{ActionReceiver, Agent, Obs};

pub struct RandomAgent {
    rng: SmallRng,
}

impl Agent for RandomAgent {
    fn act_raw(&mut self, _action: &str, num_actions: u64, _: &Obs) -> Option<u64> {
        Some(self.rng.gen_range(0..num_actions))
    }

    fn act_async_raw(&mut self, _action: &str, num_actions: u64, _: &Obs) -> ActionReceiver<u64> {
        ActionReceiver::value(self.rng.gen_range(0..num_actions))
    }
}

impl RandomAgent {
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
