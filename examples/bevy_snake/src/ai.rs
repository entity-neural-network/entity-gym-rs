use bevy::app::AppExit;
use bevy::prelude::{EventWriter, NonSendMut, Query, Res};
use entity_gym_rs::agent::{Agent, AgentOps, Featurizable, Obs};

use crate::{Direction, Position, SnakeHead, SnakeSegments};

pub(crate) fn snake_movement_agent(
    mut player: NonSendMut<Player>,
    mut heads: Query<(&mut SnakeHead, &Position)>,
    mut exit: EventWriter<AppExit>,
    segments_res: Res<SnakeSegments>,
    food: Query<(&crate::Food, &Position)>,
    segment: Query<(&crate::SnakeSegment, &Position)>,
) {
    if let Some((mut head, head_pos)) = heads.iter_mut().next() {
        let obs = Obs::new(segments_res.len() as f32 * 0.01)
            .entities(food.iter().map(|(_, p)| Food { x: p.x, y: p.y }))
            .entities([head_pos].iter().map(|p| Head { x: p.x, y: p.y }))
            .entities(segment.iter().map(|(_, p)| SnakeSegment { x: p.x, y: p.y }));
        let action = player.0.act::<Direction>(&obs);
        match action {
            Some(dir) => {
                if dir != head.direction.opposite() {
                    head.direction = dir;
                }
            }
            None => exit.send(AppExit),
        }
    }
}

pub struct Player(pub Box<dyn Agent>);

#[derive(Featurizable)]
pub struct Head {
    x: i32,
    y: i32,
}

#[derive(Featurizable)]
pub struct SnakeSegment {
    x: i32,
    y: i32,
}

#[derive(Featurizable)]
pub struct Food {
    x: i32,
    y: i32,
}
