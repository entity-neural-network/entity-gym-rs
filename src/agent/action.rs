pub trait Action {
    fn from_u64(index: u64) -> Self;
    fn to_u64(&self) -> u64;
    fn num_actions() -> u64;
    fn name() -> &'static str;
    fn labels() -> &'static [&'static str];
}

#[cfg(test)]
mod test {
    use entity_gym_derive::Action;

    use super::*;

    #[derive(Action, PartialEq, Eq, Debug)]
    enum Direction {
        Up,
        Down,
        Left,
        Right,
    }

    #[test]
    fn test_action_from_u64() {
        assert_eq!(Direction::Up.to_u64(), 0);
        assert_eq!(Direction::Down.to_u64(), 1);
        assert_eq!(Direction::Left.to_u64(), 2);
        assert_eq!(Direction::Right.to_u64(), 3);
    }

    #[test]
    fn test_action_to_u64() {
        assert_eq!(Direction::from_u64(0), Direction::Up);
        assert_eq!(Direction::from_u64(1), Direction::Down);
        assert_eq!(Direction::from_u64(2), Direction::Left);
        assert_eq!(Direction::from_u64(3), Direction::Right);
    }

    #[test]
    fn test_action_num_actions() {
        assert_eq!(Direction::num_actions(), 4);
    }

    #[test]
    fn test_action_name() {
        assert_eq!(Direction::name(), "Direction");
    }

    #[test]
    fn test_action_labels() {
        assert_eq!(Direction::labels(), &["Up", "Down", "Left", "Right"]);
    }
}
