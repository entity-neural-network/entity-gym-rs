mod ai;
#[cfg(feature = "python")]
mod python;

use std::time::Duration;

use ai::{snake_movement_agent, Player};
use bevy::app::ScheduleRunnerSettings;
use bevy::core::FixedTimestep;
use bevy::prelude::*;
use entity_gym_rs::agent::{Agent, AnyAgent};
use rand::prelude::{random, SmallRng};
use rand::{Rng, SeedableRng};

const SNAKE_HEAD_COLOR: Color = Color::rgb(0.7, 0.7, 0.7);
const FOOD_COLOR: Color = Color::rgb(1.0, 0.0, 1.0);
const SNAKE_SEGMENT_COLOR: Color = Color::rgb(0.3, 0.3, 0.3);

const ARENA_HEIGHT: u32 = 10;
const ARENA_WIDTH: u32 = 10;

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Component)]
struct Size {
    width: f32,
    height: f32,
}

impl Size {
    pub fn square(x: f32) -> Self {
        Self {
            width: x,
            height: x,
        }
    }
}

#[derive(Component)]
struct SnakeHead {
    direction: Direction,
}

struct GameOverEvent;
struct GrowthEvent;

#[derive(Default)]
struct LastTailPosition(Option<Position>);

#[derive(Component)]
struct SnakeSegment;

#[derive(Default, Deref, DerefMut)]
struct SnakeSegments(Vec<Entity>);

#[derive(Component)]
struct Food;

struct FoodTimer(u32);

#[derive(PartialEq, Copy, Clone, Debug)]
enum Direction {
    Left,
    Up,
    Right,
    Down,
}

impl Direction {
    fn opposite(self) -> Self {
        match self {
            Self::Left => Self::Right,
            Self::Right => Self::Left,
            Self::Up => Self::Down,
            Self::Down => Self::Up,
        }
    }
}

fn setup_camera(mut commands: Commands) {
    commands.spawn_bundle(OrthographicCameraBundle::new_2d());
}

fn spawn_snake(
    mut commands: Commands,
    mut segments: ResMut<SnakeSegments>,
    mut rng: ResMut<SmallRng>,
) {
    *segments = SnakeSegments(vec![commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                color: SNAKE_HEAD_COLOR,
                ..default()
            },
            ..default()
        })
        .insert(SnakeHead {
            direction: Direction::Up,
        })
        .insert(SnakeSegment)
        .insert(Position {
            x: rng.gen_range(0..ARENA_WIDTH) as i32,
            y: rng.gen_range(0..ARENA_HEIGHT) as i32,
        })
        .insert(Size::square(0.8))
        .id()]);
}

fn spawn_segment(mut commands: Commands, position: Position) -> Entity {
    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                color: SNAKE_SEGMENT_COLOR,
                ..default()
            },
            ..default()
        })
        .insert(SnakeSegment)
        .insert(position)
        .insert(Size::square(0.65))
        .id()
}

fn snake_movement(
    mut last_tail_position: ResMut<LastTailPosition>,
    mut game_over_writer: EventWriter<GameOverEvent>,
    segments: ResMut<SnakeSegments>,
    mut heads: Query<(Entity, &SnakeHead)>,
    mut positions: Query<&mut Position>,
) {
    if let Some((head_entity, head)) = heads.iter_mut().next() {
        let segment_positions = segments
            .iter()
            .map(|e| *positions.get_mut(*e).unwrap())
            .collect::<Vec<Position>>();
        let mut head_pos = positions.get_mut(head_entity).unwrap();
        match &head.direction {
            Direction::Left => {
                head_pos.x -= 1;
            }
            Direction::Right => {
                head_pos.x += 1;
            }
            Direction::Up => {
                head_pos.y += 1;
            }
            Direction::Down => {
                head_pos.y -= 1;
            }
        };
        if head_pos.x < 0
            || head_pos.y < 0
            || head_pos.x as u32 >= ARENA_WIDTH
            || head_pos.y as u32 >= ARENA_HEIGHT
        {
            game_over_writer.send(GameOverEvent);
        }
        if segment_positions.contains(&head_pos) {
            game_over_writer.send(GameOverEvent);
        }
        segment_positions
            .iter()
            .zip(segments.iter().skip(1))
            .for_each(|(pos, segment)| {
                *positions.get_mut(*segment).unwrap() = *pos;
            });
        *last_tail_position = LastTailPosition(Some(*segment_positions.last().unwrap()));
    }
}

fn snake_movement_input(keyboard_input: Res<Input<KeyCode>>, mut heads: Query<&mut SnakeHead>) {
    if let Some(mut head) = heads.iter_mut().next() {
        let dir: Direction = if keyboard_input.pressed(KeyCode::Left) {
            Direction::Left
        } else if keyboard_input.pressed(KeyCode::Down) {
            Direction::Down
        } else if keyboard_input.pressed(KeyCode::Up) {
            Direction::Up
        } else if keyboard_input.pressed(KeyCode::Right) {
            Direction::Right
        } else {
            head.direction
        };
        if dir != head.direction.opposite() {
            head.direction = dir;
        }
    }
}

fn game_over(
    mut commands: Commands,
    mut reader: EventReader<GameOverEvent>,
    mut player: ResMut<Player>,
    rng: ResMut<SmallRng>,
    segments_res: ResMut<SnakeSegments>,
    food: Query<Entity, With<Food>>,
    segments: Query<Entity, With<SnakeSegment>>,
) {
    if reader.iter().next().is_some() {
        for ent in food.iter().chain(segments.iter()) {
            commands.entity(ent).despawn();
        }
        spawn_snake(commands, segments_res, rng);
        player.0.game_over();
    }
}

fn snake_eating(
    mut commands: Commands,
    mut growth_writer: EventWriter<GrowthEvent>,
    food_positions: Query<(Entity, &Position), With<Food>>,
    head_positions: Query<&Position, With<SnakeHead>>,
) {
    for head_pos in head_positions.iter() {
        for (ent, food_pos) in food_positions.iter() {
            if food_pos == head_pos {
                commands.entity(ent).despawn();
                growth_writer.send(GrowthEvent);
            }
        }
    }
}

fn snake_growth(
    commands: Commands,
    last_tail_position: Res<LastTailPosition>,
    mut segments: ResMut<SnakeSegments>,
    mut growth_reader: EventReader<GrowthEvent>,
) {
    if growth_reader.iter().next().is_some() {
        segments.push(spawn_segment(commands, last_tail_position.0.unwrap()));
    }
}

fn size_scaling(windows: Res<Windows>, mut q: Query<(&Size, &mut Transform)>) {
    let window = windows.get_primary().unwrap();
    for (sprite_size, mut transform) in q.iter_mut() {
        transform.scale = Vec3::new(
            sprite_size.width / ARENA_WIDTH as f32 * window.width() as f32,
            sprite_size.height / ARENA_HEIGHT as f32 * window.height() as f32,
            1.0,
        );
    }
}

fn position_translation(windows: Res<Windows>, mut q: Query<(&Position, &mut Transform)>) {
    fn convert(pos: f32, bound_window: f32, bound_game: f32) -> f32 {
        let tile_size = bound_window / bound_game;
        pos / bound_game * bound_window - (bound_window / 2.) + (tile_size / 2.)
    }
    let window = windows.get_primary().unwrap();
    for (pos, mut transform) in q.iter_mut() {
        transform.translation = Vec3::new(
            convert(pos.x as f32, window.width() as f32, ARENA_WIDTH as f32),
            convert(pos.y as f32, window.height() as f32, ARENA_HEIGHT as f32),
            0.0,
        );
    }
}

fn food_spawner(mut commands: Commands, mut food_timer: ResMut<FoodTimer>) {
    if food_timer.0 == 0 {
        food_timer.0 = 7;
        commands
            .spawn_bundle(SpriteBundle {
                sprite: Sprite {
                    color: FOOD_COLOR,
                    ..default()
                },
                ..default()
            })
            .insert(Food)
            .insert(Position {
                x: (random::<f32>() * ARENA_WIDTH as f32) as i32,
                y: (random::<f32>() * ARENA_HEIGHT as f32) as i32,
            })
            .insert(Size::square(0.8));
    } else {
        food_timer.0 -= 1;
    }
}

pub fn run() {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.04, 0.04, 0.04)))
        .insert_resource(WindowDescriptor {
            title: "Snake!".to_string(),
            width: 500.0,
            height: 500.0,
            ..default()
        })
        //.insert_resource(Player::Random(RandomAgent::default()))
        .insert_resource(Player(AnyAgent::rogue_net(
            "norelattn-1m/latest-step000000999424",
        )))
        .insert_resource(FoodTimer(7))
        .insert_resource(SmallRng::seed_from_u64(0))
        .add_startup_system(setup_camera)
        .add_startup_system(spawn_snake)
        .insert_resource(SnakeSegments::default())
        .insert_resource(LastTailPosition::default())
        .add_event::<GrowthEvent>()
        .add_system(snake_movement_input.before(snake_movement))
        .add_event::<GameOverEvent>()
        .add_system_set(
            SystemSet::new()
                .with_run_criteria(FixedTimestep::step(0.150))
                .with_system(snake_movement_agent)
                .with_system(snake_movement)
                .with_system(snake_eating.after(snake_movement))
                .with_system(snake_growth.after(snake_eating))
                .with_system(food_spawner.after(snake_growth)),
        )
        .add_system(game_over.after(snake_movement))
        .add_system_set_to_stage(
            CoreStage::PostUpdate,
            SystemSet::new()
                .with_system(position_translation)
                .with_system(size_scaling),
        )
        .add_plugins(DefaultPlugins)
        .run();
}

pub fn run_headless(agent: AnyAgent, seed: u64) {
    App::new()
        .insert_resource(ScheduleRunnerSettings::run_loop(Duration::from_secs_f64(
            0.0,
        )))
        .insert_resource(Player(agent))
        .insert_resource(FoodTimer(7))
        .add_startup_system(setup_camera)
        .add_startup_system(spawn_snake)
        .insert_resource(SmallRng::seed_from_u64(seed))
        .insert_resource(SnakeSegments::default())
        .insert_resource(LastTailPosition::default())
        .add_event::<GrowthEvent>()
        //.add_system(snake_movement_input.before(snake_movement))
        .add_event::<GameOverEvent>()
        .add_system_set(
            SystemSet::new()
                .with_system(snake_movement_agent)
                .with_system(snake_movement.after(snake_movement_agent))
                .with_system(snake_eating.after(snake_movement))
                .with_system(snake_growth.after(snake_eating))
                .with_system(food_spawner.after(snake_growth)),
        )
        .add_system(game_over.after(snake_movement))
        .add_plugins(MinimalPlugins)
        .run();
}
