use std::collections::HashMap;

use super::Featurizable;

pub struct Obs {
    pub(crate) entities: HashMap<String, (Vec<f32>, usize)>,
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

    pub fn entities<E: Featurizable, I: Iterator<Item = E>>(
        mut self,
        name: &str,
        entities: I,
    ) -> Self {
        let mut feats = vec![];
        let mut count = 0;
        for entity in entities {
            feats.extend(entity.featurize());
            count += 1;
        }
        self.entities.insert(name.to_string(), (feats, count));
        self
    }

    pub fn score(&self) -> f32 {
        self.score
    }
}
