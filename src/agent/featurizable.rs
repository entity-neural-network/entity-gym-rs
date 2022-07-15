pub trait Featurizable {
    fn num_feats() -> usize;
    fn feature_names() -> Vec<String>;
    fn featurize(&self) -> Vec<f32>;
}

impl<'a, T: Featurizable> Featurizable for &'a T {
    fn num_feats() -> usize {
        T::num_feats()
    }

    fn feature_names() -> Vec<String> {
        T::feature_names()
    }

    fn featurize(&self) -> Vec<f32> {
        (*self).featurize()
    }
}
