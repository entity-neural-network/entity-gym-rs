use bevy::app::AppExit;
use bevy::prelude::{EventWriter, NonSendMut, Query, Res};
use entity_gym_rs::agent::{Action, Agent, AnyAgent, Featurizable, Obs};
use entity_gym_rs::Featurizable;

use crate::{Direction, Pause, Player, Position, SnakeHead, SnakeSegments};

pub(crate) fn snake_movement_agent(
    mut players: NonSendMut<Players>,
    mut heads: Query<(&mut SnakeHead, &Position, &Player)>,
    mut exit: EventWriter<AppExit>,
    pause: Res<Pause>,
    segments_res: Res<SnakeSegments>,
    food: Query<(&crate::Food, &Position)>,
    segment: Query<(&crate::SnakeSegment, &Position, &Player)>,
) {
    if pause.0 > 0 {
        return;
    }
    let mut head_actions = vec![];
    for (head, head_pos, player) in heads.iter_mut() {
        if let Some(agent) = &mut players.0[player.index()] {
            let obs = Obs::new(segments_res.0[player.index()].len() as f32)
                .entities(food.iter().map(|(_, p)| Food { x: p.x, y: p.y }))
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
            let action = agent.act_async::<Move>(&obs);
            head_actions.push((head, action));
        }
    }
    for (mut head, action) in head_actions.into_iter() {
        match action.rcv() {
            Some(Move(dir)) => {
                if dir != head.direction.opposite() {
                    head.direction = dir;
                }
            }
            None => exit.send(AppExit),
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

#[derive(Featurizable)]
pub struct Head {
    x: i32,
    y: i32,
    is_enemy: bool,
}

#[derive(Featurizable)]
pub struct SnakeSegment {
    x: i32,
    y: i32,
    is_enemy: bool,
}

#[derive(Featurizable)]
pub struct Food {
    x: i32,
    y: i32,
}
