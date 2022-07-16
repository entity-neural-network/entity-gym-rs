use std::collections::HashMap;

use super::Featurizable;

pub struct Obs {
    pub(crate) entities: HashMap<&'static str, (Vec<f32>, usize, usize)>,
    pub(crate) done: bool,
    pub(crate) score: f32,
}

impl Obs {
    pub fn new(score: f32) -> Self {
        Obs {
            score,
            entities: Default::default(),
            done: false,
        }
    }

    pub fn entities<E: Featurizable, I: Iterator<Item = E>>(mut self, entities: I) -> Self {
        let mut feats = vec![];
        let mut count = 0;
        for entity in entities {
            feats.extend(entity.featurize());
            count += 1;
        }
        self.entities
            .insert(E::name(), (feats, count, E::num_feats()));
        self
    }

    pub fn score(&self) -> f32 {
        self.score
    }
}
