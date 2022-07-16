pub trait Featurizable {
    fn num_feats() -> usize;
    fn feature_names() -> &'static [&'static str];
    fn featurize(&self) -> Vec<f32>;
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
