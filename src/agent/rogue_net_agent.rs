use ndarray::Array2;
use rogue_net::RogueNet;

use super::Obs;
use super::{ActionReceiver, Agent};

#[derive(Clone)]
pub struct RogueNetAgent {
    pub(crate) net: RogueNet,
}

impl RogueNetAgent {
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Self {
        RogueNetAgent {
            net: RogueNet::load(path),
        }
    }
}

impl Agent for RogueNetAgent {
    fn act_raw(&mut self, _action: &str, _num_actions: u64, obs: &Obs) -> Option<u64> {
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
        Some(acts[0])
    }

    fn act_async_raw(&mut self, action: &str, num_actions: u64, obs: &Obs) -> ActionReceiver<u64> {
        ActionReceiver::value(self.act_raw(action, num_actions, obs).unwrap())
    }
}
