/// A data structure that can be serialized into a data format that can be processed by a neural network.
///
/// Can be derived for structs where all fields are numeric, boolean, or [`Featurizable`].
///
/// # Example
/// ```rust
/// use entity_gym_rs::agent::Featurizable;
///
/// #[derive(Featurizable)]
/// struct Player { x: i32, y: i32, is_alive: bool }
/// ```
pub trait Featurizable {
    /// Returns the number of features after conversion to a vector.
    fn num_feats() -> usize;
    /// Returns a list of human readable labels corresponding to each feature.
    fn feature_names() -> &'static [&'static str];
    /// Serializes the entity into a vector of features.
    fn featurize(&self) -> Vec<f32>;
    /// Returns a human readable name for the entity.
    fn name() -> &'static str;
}

impl<'a, T: Featurizable> Featurizable for &'a T {
    fn num_feats() -> usize {
        T::num_feats()
    }

    fn feature_names() -> &'static [&'static str] {
        T::feature_names()
    }

    fn featurize(&self) -> Vec<f32> {
        (*self).featurize()
    }

    fn name() -> &'static str {
        T::name()
    }
}

mod test {
    use super::*;

    use entity_gym_derive::Featurizable;

    #[derive(Featurizable)]
    struct Pos {
        x: f32,
        y: f32,
    }

    #[derive(Featurizable)]
    struct Hero {
        x: f32,
        y: f32,
        level: u32,
        alive: bool,
    }

    #[test]
    fn test_num_feats() {
        assert_eq!(Pos::num_feats(), 2);
        assert_eq!(Hero::num_feats(), 4);
    }

    #[test]
    fn test_feature_names() {
        assert_eq!(Pos::feature_names(), &["x", "y"]);
        assert_eq!(Hero::feature_names(), &["x", "y", "level", "alive"]);
    }

    #[test]
    fn test_featurize() {
        assert_eq!(Pos::featurize(&Pos { x: 1.0, y: 2.0 }), vec![1.0, 2.0]);
        assert_eq!(
            Hero::featurize(&Hero {
                x: 1.0,
                y: 2.0,
                level: 3,
                alive: true,
            }),
            vec![1.0, 2.0, 3.0, 1.0]
        );
    }

    #[test]
    fn test_name() {
        assert_eq!(Pos::name(), "Pos");
        assert_eq!(Hero::name(), "Hero");
    }
}
