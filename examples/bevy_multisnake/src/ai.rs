use bevy::app::AppExit;
use bevy::prelude::{EventWriter, NonSendMut, Query, Res};
use entity_gym_rs::agent::{Action, Agent, AnyAgent, Featurizable, Obs};

use crate::{Direction, Player, Position, SnakeHead, SnakeSegments};

pub(crate) fn snake_movement_agent(
    mut players: NonSendMut<Players>,
    mut heads: Query<(&mut SnakeHead, &Position, &Player)>,
    mut exit: EventWriter<AppExit>,
    segments_res: Res<SnakeSegments>,
    food: Query<(&crate::Food, &Position, &Player)>,
    segment: Query<(&crate::SnakeSegment, &Position, &Player)>,
) {
    for (mut head, head_pos, player) in heads.iter_mut() {
        if let Some(agent) = &mut players.0[player.index()] {
            let obs = Obs::new(segments_res.len() as f32)
                .entities(food.iter().map(|(_, p, plr)| Food {
                    x: p.x,
                    y: p.y,
                    is_enemy: player != plr,
                }))
                .entities([head_pos].iter().map(|p| Head {
                    x: p.x,
                    y: p.y,
                    is_enemy: false,
                }))
                .entities(segment.iter().map(|(_, p, plr)| SnakeSegment {
                    x: p.x,
                    y: p.y,
                    is_enemy: player != plr,
                }));
            let action = agent.act::<Move>(&obs);
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
}
pub struct Players(pub [Option<AnyAgent>; 2]);

pub struct Move(Direction);

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

    fn name() -> &'static str {
        "move"
    }

    fn labels() -> &'static [&'static str] {
        &["up", "down", "left", "right"]
    }
}

pub struct Head {
    x: i32,
    y: i32,
    is_enemy: bool,
}

impl Featurizable for Head {
    fn num_feats() -> usize {
        3
    }

    fn feature_names() -> &'static [&'static str] {
        &["x", "y", "is_enemy"]
    }

    fn featurize(&self) -> Vec<f32> {
        vec![
            self.x as f32,
            self.y as f32,
            if self.is_enemy { 1. } else { 0. },
        ]
    }

    fn name() -> &'static str {
        "Head"
    }
}

pub struct SnakeSegment {
    x: i32,
    y: i32,
    is_enemy: bool,
}

impl Featurizable for SnakeSegment {
    fn num_feats() -> usize {
        3
    }

    fn feature_names() -> &'static [&'static str] {
        &["x", "y", "is_enemy"]
    }

    fn featurize(&self) -> Vec<f32> {
        vec![
            self.x as f32,
            self.y as f32,
            if self.is_enemy { 1. } else { 0. },
        ]
    }

    fn name() -> &'static str {
        "SnakeSegment"
    }
}

pub struct Food {
    x: i32,
    y: i32,
    is_enemy: bool,
}

impl Featurizable for Food {
    fn num_feats() -> usize {
        3
    }

    fn feature_names() -> &'static [&'static str] {
        &["x", "y", "is_enemy"]
    }

    fn featurize(&self) -> Vec<f32> {
        vec![
            self.x as f32,
            self.y as f32,
            if self.is_enemy { 1. } else { 0. },
        ]
    }

    fn name() -> &'static str {
        "Food"
    }
}
