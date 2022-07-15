use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

use super::{Action, Agent, Obs};

pub struct RandomAgent {
    rng: SmallRng,
}

impl Agent for RandomAgent {
    fn act<A: Action>(&mut self, _: Obs) -> Option<A> {
        Some(A::from_u64(self.rng.gen_range(0..A::num_actions())))
    }
}

impl RandomAgent {
    fn new(seed: u64) -> Self {
        RandomAgent {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Default for RandomAgent {
    fn default() -> Self {
        RandomAgent::new(0)
    }
}
