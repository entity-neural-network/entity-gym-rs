pub trait Action {
    fn from_u64(index: u64) -> Self;
    fn to_u64(&self) -> u64;
    fn num_actions() -> u64;
}
