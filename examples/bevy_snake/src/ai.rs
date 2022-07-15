use bevy::app::AppExit;
use bevy::prelude::{Component, EventWriter, Query, Res, ResMut};
use entity_gym_rs::agent::{Action, Agent, AnyAgent, Featurizable, Obs};

use crate::{Direction, Food, Position, SnakeHead, SnakeSegment, SnakeSegments};

pub(crate) fn snake_movement_agent(
    mut player: ResMut<Player>,
    mut heads: Query<(&mut SnakeHead, &Position)>,
    mut exit: EventWriter<AppExit>,
    segments_res: Res<SnakeSegments>,
    food: Query<(&Food, &Position)>,
    segment: Query<(&SnakeSegment, &Position)>,
) {
    if let Some((mut head, head_pos)) = heads.iter_mut().next() {
        let obs = Obs::new(segments_res.len() as f32)
            .entities("Food", food.iter().map(|(_, pos)| pos))
            .entities("Head", [head_pos].into_iter())
            .entities("SnakeSegment", segment.iter().map(|(_, pos)| pos));
        let action = player.0.act::<Move>(obs);
        match action {
            Some(Move(dir)) => {
                if dir != head.direction.opposite() {
                    head.direction = dir;
                }
            }
            None => exit.send(AppExit),
        }
    }
}

#[derive(Component)]
pub struct Player(pub AnyAgent);

struct Move(Direction);

impl Action for Move {
    fn from_u64(index: u64) -> Self {
        match index {
            0 => Move(Direction::Up),
            1 => Move(Direction::Down),
            2 => Move(Direction::Left),
            3 => Move(Direction::Right),
            _ => panic!("Invalid direction index"),
        }
    }

    fn to_u64(&self) -> u64 {
        match self.0 {
            Direction::Up => 0,
            Direction::Down => 1,
            Direction::Left => 2,
            Direction::Right => 3,
        }
    }

    fn num_actions() -> u64 {
        4
    }
}

impl Featurizable for Position {
    fn num_feats() -> usize {
        2
    }

    fn feature_names() -> Vec<String> {
        vec!["x".to_string(), "y".to_string()]
    }

    fn featurize(&self) -> Vec<f32> {
        vec![self.x as f32, self.y as f32]
    }
}
