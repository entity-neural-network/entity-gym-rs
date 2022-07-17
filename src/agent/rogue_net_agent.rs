use ndarray::Array2;
use rogue_net::{load_checkpoint, RogueNet};

use super::{Action, Obs};
use super::{ActionReceiver, Agent};

pub struct RogueNetAgent {
    net: RogueNet,
}

impl RogueNetAgent {
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Self {
        let net = load_checkpoint(path);
        RogueNetAgent { net }
    }
}

impl Agent for RogueNetAgent {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A> {
        let entities = obs
            .entities
            .iter()
            .map(|(name, (feats, count, dim))| {
                (
                    name.to_string(),
                    Array2::from_shape_vec((*count, *dim), feats.clone()).unwrap(),
                )
            })
            .collect();
        let (_probs, acts) = self.net.forward(&entities);
        Some(A::from_u64(acts[0]))
    }

    fn act_async<A: Action>(&mut self, obs: &Obs) -> ActionReceiver<A> {
        ActionReceiver::value(self.act(obs).unwrap())
    }
}
