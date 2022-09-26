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
    fn labels() -> Vec<String>;
}

mod expand {
    use entity_gym_derive::Action;

    use super::*;

    #[derive(Action, PartialEq, Eq, Debug)]
    enum Direction {
        Up,
        Down,
        Left,
        Right,
    }

    #[derive(Action, PartialEq, Eq, Debug)]
    enum Thrust {
        Full,
        Half,
        None,
    }

    #[derive(Action, PartialEq, Eq, Debug)]
    struct Move {
        direction: Direction,
        thrust: Thrust,
    }
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

    #[derive(Action, PartialEq, Eq, Debug, Clone, Copy)]
    enum Thrust {
        Full,
        Half,
        None,
    }

    #[derive(Action, PartialEq, Eq, Debug)]
    struct Move {
        direction: Direction,
        thrust: Thrust,
    }

    #[test]
    fn test_action_from_u64() {
        assert_eq!(Direction::Up.to_u64(), 0);
        assert_eq!(Direction::Down.to_u64(), 1);
        assert_eq!(Direction::Left.to_u64(), 2);
        assert_eq!(Direction::Right.to_u64(), 3);
        assert_eq!(
            Move {
                direction: Direction::Up,
                thrust: Thrust::Full
            }
            .to_u64(),
            0
        );
        assert_eq!(
            Move {
                direction: Direction::Up,
                thrust: Thrust::Half
            }
            .to_u64(),
            1,
        );
        assert_eq!(
            Move {
                direction: Direction::Up,
                thrust: Thrust::None
            }
            .to_u64(),
            2,
        );
        assert_eq!(
            Move {
                direction: Direction::Down,
                thrust: Thrust::Full
            }
            .to_u64(),
            3,
        );
        assert_eq!(
            Move {
                direction: Direction::Down,
                thrust: Thrust::Half
            }
            .to_u64(),
            4
        );
        assert_eq!(
            Move {
                direction: Direction::Down,
                thrust: Thrust::None
            }
            .to_u64(),
            5
        );
    }

    #[test]
    fn test_action_to_u64() {
        assert_eq!(Direction::from_u64(0), Direction::Up);
        assert_eq!(Direction::from_u64(1), Direction::Down);
        assert_eq!(Direction::from_u64(2), Direction::Left);
        assert_eq!(Direction::from_u64(3), Direction::Right);
        assert_eq!(
            Move::from_u64(0),
            Move {
                direction: Direction::Up,
                thrust: Thrust::Full
            }
        );
        assert_eq!(
            Move::from_u64(1),
            Move {
                direction: Direction::Up,
                thrust: Thrust::Half
            }
        );
        assert_eq!(
            Move::from_u64(7),
            Move {
                direction: Direction::Left,
                thrust: Thrust::Half
            }
        );
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
        assert_eq!(
            Move::labels(),
            &[
                "direction=Up,thrust=Full",
                "direction=Up,thrust=Half",
                "direction=Up,thrust=None",
                "direction=Down,thrust=Full",
                "direction=Down,thrust=Half",
                "direction=Down,thrust=None",
                "direction=Left,thrust=Full",
                "direction=Left,thrust=Half",
                "direction=Left,thrust=None",
                "direction=Right,thrust=Full",
                "direction=Right,thrust=Half",
                "direction=Right,thrust=None",
            ]
        );
    }

    #[test]
    fn test_round_trip() {
        for thrust in [Thrust::Full, Thrust::Half, Thrust::None] {
            for direction in vec![
                Direction::Up,
                Direction::Down,
                Direction::Left,
                Direction::Right,
            ] {
                let action = Move { direction, thrust };
                let index = action.to_u64();
                let action2 = Move::from_u64(index);
                assert_eq!(action, action2);
            }
        }
    }
}
