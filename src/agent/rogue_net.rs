use ndarray::Array2;
use rogue_net::RogueNet;

use super::{ActionReceiver, Agent};
use super::{Featurizable, Obs};

/// Agent that implements the [RogueNet entity neural network](https://github.com/entity-neural-network/rogue-net).
/// Can be loaded from checkpoints produced by [enn-trainer](https://github.com/entity-neural-network/enn-trainer).
#[derive(Clone)]
pub struct RogueNetAgent {
    pub(crate) net: RogueNet,
}

impl RogueNetAgent {
    /// Loads a neural network agent from an [enn-trainer](https://github.com/entity-neural-network/enn-trainer) checkpoint directory.
    pub fn load<P: AsRef<std::path::Path>>(path: P) -> Self {
        RogueNetAgent {
            net: RogueNet::load(path),
        }
    }

    /// Loads a neural network agent from an archive of a checkpoint directory.
    ///
    /// You can use the rogue-net cli to create a tar archive of a checkpoint directory:
    /// ```console
    /// $ cargo install rogue-net
    /// $ rogue-net archive --path path/to/checkpoint/dir
    /// ```
    pub fn load_archive<R: std::io::Read>(reader: R) -> Result<Self, std::io::Error> {
        let net = RogueNet::load_archive(reader)?;
        Ok(RogueNetAgent { net })
    }

    /// Adapts the network to a changed observation space.
    ///
    /// If you trained a network with a different observation space, you can adapt it to the new observation space.
    /// For this to work, the new set of features of the observation space must be a superset of the old set of features.
    pub fn with_feature_adaptor<E: Featurizable>(mut self) -> Self {
        self.net = self.net.with_obs_filter(
            [(E::name().to_string(), E::feature_names())]
                .iter()
                .cloned()
                .collect(),
        );
        self
    }
}

impl Agent for RogueNetAgent {
    fn act_dyn(&mut self, _action: &str, _num_actions: u64, obs: &Obs) -> Option<u64> {
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

    fn act_async_dyn(&mut self, action: &str, num_actions: u64, obs: &Obs) -> ActionReceiver<u64> {
        ActionReceiver::value(self.act_dyn(action, num_actions, obs).unwrap())
    }

    fn game_over(&mut self, _: &Obs) {}
}
