/// Defines a categorical action. Can be derived for enums.
///
/// # Example
/// ```rust
/// use entity_gym_rs::agent::Action;
///
/// #[derive(Action)]
/// enum Move { Up, Down, Left, Right }
/// ```
pub trait Action<'a> {
    /// Instantiates an action from a u64.
    fn from_u64(index: u64) -> Self;
    /// Converts an action to a u64.
    fn to_u64(&self) -> u64;
    /// Returns the number of possible action choices.
    fn num_actions() -> u64;
    /// Returns the human readable name of the action.
    fn name() -> &'a str;
    /// Returns a list of human readable labels corresponding to each action choice.
    fn labels() -> &'a [&'a str];
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
