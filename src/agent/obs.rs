use std::collections::HashMap;

use rustc_hash::FxHashMap;

use super::Featurizable;

/// An observation that defines what an agent can see.
///
/// The `Obs::new` method creates an observation.
/// It takes a single required parameter, the current score, which is maximized during training.
///
/// ```rust
/// use entity_gym_rs::agent::Obs;
///
/// let obs = Obs::new(0.0);
/// ```
///
/// An observation can contain any number of `Featurizable` entities.
///
/// ```rust
/// use entity_gym_rs::agent::{Obs, Featurizable};
///
/// #[derive(Featurizable)]
/// struct Player { x: i32, y: i32 }
///
/// #[derive(Featurizable)]
/// struct Cake { is_a_lie: bool }
///
/// let obs = Obs::new(0.0)
///     .entities([Player { x: 0, y: -3 }])
///     .entities([Cake { is_a_lie: true }, Cake { is_a_lie: false }]);
/// ```
pub struct Obs {
    pub(crate) entities: HashMap<&'static str, (Vec<f32>, usize, usize)>,
    // Field is only accessed when cfg(feature = "python").
    #[allow(dead_code)]
    pub(crate) done: bool,
    pub(crate) score: f32,
    pub(crate) metrics: FxHashMap<String, f32>,
}

impl Obs {
    /// Creates a new observation.
    ///
    /// # Arguments
    /// * `score` - The current score of the agent. This is maximized during training.
    pub fn new(score: f32) -> Self {
        Obs {
            score,
            entities: Default::default(),
            done: false,
            metrics: Default::default(),
        }
    }

    /// Adds a set of entities to the observation.
    ///
    /// # Arguments
    /// * `entities` - An iterator of [`Featurizable`] entities to add to the observation.
    pub fn entities<E: Featurizable, I: IntoIterator<Item = E>>(mut self, entities: I) -> Self {
        let mut feats = vec![];
        let mut count = 0;
        for entity in entities.into_iter() {
            feats.extend(entity.featurize());
            count += 1;
        }
        self.entities
            .insert(E::name(), (feats, count, E::num_feats()));
        self
    }

    /// Adds a numerical metric to the observation. Aggregate statistics of all metrics are surfaced during training.
    ///
    /// # Arguments
    /// * `name` - The name of the metric.
    /// * `value` - The value of the metric.
    ///
    /// # Example
    /// ```rust
    /// use entity_gym_rs::agent::Obs;
    ///
    /// let mut obs = Obs::new(0.0)
    ///     .metric("experience-collected", 2132.4)
    ///     .metric("game-over-reason/touched-enemy", 1.0);
    /// ```
    pub fn metric(mut self, name: &str, value: f32) -> Self {
        self.metrics.insert(name.to_string(), value);
        self
    }

    /// Returns the score.
    pub fn score(&self) -> f32 {
        self.score
    }
}
