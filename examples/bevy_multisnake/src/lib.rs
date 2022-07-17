mod ai;
#[cfg(feature = "python")]
pub mod python;

use std::time::Duration;

use ai::{snake_movement_agent, Players};
use bevy::app::ScheduleRunnerSettings;
use bevy::core::FixedTimestep;
use bevy::prelude::*;
use entity_gym_rs::agent::{Agent, AnyAgent};
use rand::prelude::{random, SmallRng};
use rand::{Rng, SeedableRng};

const HEAD_COLOR: [Color; 2] = [Color::rgb(0.6, 0.6, 1.0), Color::rgb(1.0, 0.6, 0.6)];
const FOOD_COLOR: [Color; 2] = [Color::rgb(0.0, 0.0, 1.0), Color::rgb(1.0, 0.0, 0.0)];
const SNAKE_SEGMENT_COLOR: [Color; 2] = [Color::rgb(0.2, 0.2, 0.4), Color::rgb(0.4, 0.2, 0.2)];

const ARENA_HEIGHT: u32 = 10;
const ARENA_WIDTH: u32 = 10;

struct Config {
    easy_mode: bool,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
struct Position {
    x: i32,
    y: i32,
}

#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum Player {
    Blue,
    Red,
}

impl Player {
    fn index(&self) -> usize {
        match self {
            Player::Blue => 0,
            Player::Red => 1,
        }
    }
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
struct GrowthEvent(Player);

#[derive(Default)]
struct LastTailPosition([Option<Position>; 2]);

#[derive(Component)]
struct SnakeSegment;

#[derive(Default, Deref, DerefMut)]
struct SnakeSegments([Vec<Entity>; 2]);

#[derive(Component)]
struct Food;

struct FoodTimer([Option<u32>; 2]);

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
    let mut ss = (0..2)
        .map(|p| {
            vec![commands
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        color: HEAD_COLOR[p],
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
                .insert(if p == 0 { Player::Blue } else { Player::Red })
                .insert(Size::square(0.8))
                .id()]
        })
        .collect::<Vec<_>>();
    let s2 = ss.pop().unwrap();
    let s1 = ss.pop().unwrap();
    *segments = SnakeSegments([s1, s2]);
}

fn spawn_segment(commands: &mut Commands, position: Position, player: Player) -> Entity {
    commands
        .spawn_bundle(SpriteBundle {
            sprite: Sprite {
                color: SNAKE_SEGMENT_COLOR[player.index()],
                ..default()
            },
            ..default()
        })
        .insert(SnakeSegment)
        .insert(position)
        .insert(Size::square(0.65))
        .insert(player)
        .id()
}

fn snake_movement(
    config: Res<Config>,
    mut last_tail_position: ResMut<LastTailPosition>,
    mut game_over_writer: EventWriter<GameOverEvent>,
    segments: ResMut<SnakeSegments>,
    mut heads: Query<(Entity, &SnakeHead, &Player)>,
    mut positions: Query<&mut Position>,
) {
    for (head_entity, head, p) in heads.iter_mut() {
        let segment_positions = segments[p.index()]
            .iter()
            .map(|e| *positions.get_mut(*e).unwrap())
            .collect::<Vec<Position>>();
        let mut head_pos = positions.get_mut(head_entity).unwrap();
        if config.easy_mode
            && ((head_pos.x == 0 && head.direction == Direction::Left)
                || (head_pos.x == ARENA_WIDTH as i32 - 1 && head.direction == Direction::Right)
                || (head_pos.y == 0 && head.direction == Direction::Down)
                || (head_pos.y == ARENA_HEIGHT as i32 - 1 && head.direction == Direction::Up))
        {
            continue;
        }
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
            .zip(segments[p.index()].iter().skip(1))
            .for_each(|(pos, segment)| {
                *positions.get_mut(*segment).unwrap() = *pos;
            });
        last_tail_position.0[p.index()] = Some(*segment_positions.last().unwrap());
    }
}

fn snake_movement_input(keyboard_input: Res<Input<KeyCode>>, mut heads: Query<&mut SnakeHead>) {
    if let Some(mut head) = heads.iter_mut().next() {
        let dir: Direction =
            if keyboard_input.pressed(KeyCode::Left) || keyboard_input.pressed(KeyCode::A) {
                Direction::Left
            } else if keyboard_input.pressed(KeyCode::Down) || keyboard_input.pressed(KeyCode::S) {
                Direction::Down
            } else if keyboard_input.pressed(KeyCode::Up) || keyboard_input.pressed(KeyCode::W) {
                Direction::Up
            } else if keyboard_input.pressed(KeyCode::Right) || keyboard_input.pressed(KeyCode::D) {
                Direction::Right
            } else {
                head.direction
            };
        if dir != head.direction.opposite() {
            head.direction = dir;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn game_over(
    mut commands: Commands,
    mut reader: EventReader<GameOverEvent>,
    mut players: NonSendMut<Players>,
    mut food_timer: ResMut<FoodTimer>,
    rng: ResMut<SmallRng>,
    segments_res: ResMut<SnakeSegments>,
    food: Query<Entity, With<Food>>,
    segments: Query<Entity, With<SnakeSegment>>,
) {
    if reader.iter().next().is_none() {
        return;
    }
    for ent in food.iter().chain(segments.iter()) {
        commands.entity(ent).despawn();
    }
    food_timer.0 = [Some(4), Some(4)];
    for (i, player) in players.0.iter_mut().enumerate() {
        if let Some(player) = player {
            player.game_over(segments_res.0[i].len() as f32);
        }
    }
    spawn_snake(commands, segments_res, rng);
}

fn snake_eating(
    mut food_timer: ResMut<FoodTimer>,
    mut commands: Commands,
    mut growth_writer: EventWriter<GrowthEvent>,
    mut game_over_writer: EventWriter<GameOverEvent>,
    food_positions: Query<(Entity, &Position, &Player), With<Food>>,
    head_positions: Query<(&Position, &Player), With<SnakeHead>>,
) {
    for (head_pos, &player) in head_positions.iter() {
        for (ent, food_pos, player_food) in food_positions.iter() {
            if food_pos == head_pos {
                if player == *player_food {
                    commands.entity(ent).despawn();
                    growth_writer.send(GrowthEvent(player));
                    food_timer.0[player.index()] = Some(4);
                } else {
                    game_over_writer.send(GameOverEvent);
                }
            }
        }
    }
}

fn snake_growth(
    mut commands: Commands,
    last_tail_position: Res<LastTailPosition>,
    mut segments: ResMut<SnakeSegments>,
    mut growth_reader: EventReader<GrowthEvent>,
    mut game_over_writer: EventWriter<GameOverEvent>,
) {
    for growth in growth_reader.iter() {
        let player = growth.0;
        segments[player.index()].push(spawn_segment(
            &mut commands,
            last_tail_position.0[player.index()].unwrap(),
            player,
        ));
        if segments[player.index()].len() == 11 {
            game_over_writer.send(GameOverEvent);
        }
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
    for (timer, player) in food_timer.0.iter_mut().zip([Player::Blue, Player::Red]) {
        if let Some(time) = timer {
            if *time == 0 {
                commands
                    .spawn_bundle(SpriteBundle {
                        sprite: Sprite {
                            color: FOOD_COLOR[player.index()],
                            ..default()
                        },
                        ..default()
                    })
                    .insert(Food)
                    .insert(Position {
                        x: (random::<f32>() * ARENA_WIDTH as f32) as i32,
                        y: (random::<f32>() * ARENA_HEIGHT as f32) as i32,
                    })
                    .insert(player)
                    .insert(Size::square(0.8));
                *timer = None;
            } else {
                *time -= 1;
            }
        }
    }
}

pub fn run(agent_path: Option<String>, easy_mode: bool) {
    App::new()
        .insert_resource(ClearColor(Color::rgb(0.04, 0.04, 0.04)))
        .insert_resource(WindowDescriptor {
            title: "Snake!".to_string(),
            width: 500.0,
            height: 500.0,
            ..default()
        })
        .insert_resource(Config { easy_mode })
        .insert_non_send_resource(match agent_path {
            Some(path) => Players([None, Some(AnyAgent::rogue_net(&path))]),
            None => Players([None, Some(AnyAgent::random_seeded(1))]),
        })
        .insert_resource(FoodTimer([Some(4), Some(4)]))
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

pub fn run_headless(agents: [AnyAgent; 2], seed: u64) {
    let [a1, a2] = agents;
    App::new()
        .insert_resource(ScheduleRunnerSettings::run_loop(Duration::from_secs_f64(
            0.0,
        )))
        .insert_resource(Config { easy_mode: false })
        .insert_non_send_resource(Players([Some(a1), Some(a2)]))
        .insert_resource(FoodTimer([Some(4), Some(4)]))
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
