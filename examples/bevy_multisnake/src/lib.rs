mod ai;
#[cfg(feature = "python")]
pub mod python;

use std::time::Duration;

use ai::{snake_movement_agent, Players};
use bevy::app::ScheduleRunnerSettings;
use bevy::core::FixedTimestep;
use bevy::prelude::*;
use entity_gym_rs::agent::{Action, Agent, RandomAgent, RogueNetAgent};
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

const HEAD_COLOR: [Color; 2] = [Color::rgb(0.6, 0.6, 1.0), Color::rgb(1.0, 0.6, 0.6)];
const FOOD_COLOR: Color = Color::rgb(1.0, 0.0, 1.0);
const SNAKE_SEGMENT_COLOR: [Color; 2] = [Color::rgb(0.2, 0.2, 0.4), Color::rgb(0.4, 0.2, 0.2)];
const BACKGROUND_COLOR: Color = Color::rgb(0.04, 0.04, 0.04);

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

#[derive(Component, Clone, Copy, PartialEq, Eq, Debug)]
enum Player {
    Blue,
    Red,
}

struct Pause(usize);

impl Player {
    fn index(&self) -> usize {
        match self {
            Player::Blue => 0,
            Player::Red => 1,
        }
    }

    fn opponent(&self) -> Player {
        match self {
            Player::Blue => Player::Red,
            Player::Red => Player::Blue,
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

struct GameOverEvent(Option<Player>, GameOverReason);

enum GameOverReason {
    HeadCollision,
    MaxLengthReached,
    SnakeCollision,
    WallCollision,
}

struct GrowthEvent(Player);

#[derive(Default)]
struct LastTailPosition([Option<Position>; 2]);

#[derive(Component)]
struct SnakeSegment;

#[derive(Default, Deref, DerefMut)]
struct SnakeSegments([Vec<Entity>; 2]);

#[derive(Component)]
struct Food;

struct FoodTimer(Option<u32>);

#[derive(PartialEq, Copy, Clone, Debug, Action)]
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
            let half_width = ARENA_WIDTH / 2;
            let player = if p == 0 { Player::Blue } else { Player::Red };
            let x = rng.gen_range((half_width * p)..(half_width * (p + 1))) as i32;
            let y = rng.gen_range(1..(ARENA_HEIGHT - 1)) as i32;
            vec![
                commands
                    .spawn_bundle(SpriteBundle {
                        sprite: Sprite {
                            color: HEAD_COLOR[p as usize],
                            ..default()
                        },
                        ..default()
                    })
                    .insert(SnakeHead {
                        direction: Direction::Up,
                    })
                    .insert(SnakeSegment)
                    .insert(Position { x, y })
                    .insert(player)
                    .insert(Size::square(0.8))
                    .id(),
                spawn_segment(&mut commands, Position { x, y: y - 1 }, player),
            ]
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
    mut pause: ResMut<Pause>,
    segments: ResMut<SnakeSegments>,
    mut heads: Query<(Entity, &SnakeHead, &Player)>,
    mut positions: Query<&mut Position>,
) {
    if pause.0 > 0 {
        pause.0 -= 1;
        return;
    }
    #[allow(clippy::needless_collect)]
    let all_segment_positions = segments
        .iter()
        .flatten()
        .map(|e| *positions.get_mut(*e).unwrap())
        .collect::<Vec<Position>>();
    let mut head_positions = vec![];
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
            game_over_writer.send(GameOverEvent(
                Some(p.opponent()),
                GameOverReason::WallCollision,
            ));
        }
        if all_segment_positions.contains(&head_pos) {
            game_over_writer.send(GameOverEvent(
                Some(p.opponent()),
                GameOverReason::SnakeCollision,
            ));
        }
        head_positions.push(*head_pos);
        segment_positions
            .iter()
            .zip(segments[p.index()].iter().skip(1))
            .for_each(|(pos, segment)| {
                *positions.get_mut(*segment).unwrap() = *pos;
            });
        last_tail_position.0[p.index()] = Some(*segment_positions.last().unwrap());
    }
    if head_positions[0] == head_positions[1] {
        game_over_writer.send(GameOverEvent(None, GameOverReason::HeadCollision));
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
    mut clear_color: ResMut<ClearColor>,
    mut pause: ResMut<Pause>,
    rng: ResMut<SmallRng>,
    segments_res: ResMut<SnakeSegments>,
    food: Query<Entity, With<Food>>,
    segments: Query<Entity, With<SnakeSegment>>,
) {
    if let Some(GameOverEvent(winner, reason)) = reader.iter().next() {
        let game_over_reason_str = match reason {
            GameOverReason::WallCollision => "game_over/wall_collision",
            GameOverReason::SnakeCollision => "game_over/snake_collision",
            GameOverReason::HeadCollision => "game_over/head_collision",
            GameOverReason::MaxLengthReached => "game_over/max_length_reached",
        };
        pause.0 = 5;
        clear_color.0 = match winner {
            Some(winner) => SNAKE_SEGMENT_COLOR[winner.index()],
            None => BACKGROUND_COLOR,
        };
        for ent in food.iter().chain(segments.iter()) {
            commands.entity(ent).despawn();
        }
        food_timer.0 = Some(4);
        for (i, player) in players.0.iter_mut().enumerate() {
            if let Some(player) = player {
                let score = match winner {
                    Some(winner) if winner.index() == i => {
                        segments_res.0[i].len() as f32 * 0.1 + 1.0
                    }
                    _ => segments_res.0[i].len() as f32 * 0.1,
                };
                let metrics = vec![
                    (game_over_reason_str.to_string(), 1.0),
                    ("final_length".to_string(), segments_res.0[i].len() as f32),
                ];
                player.game_over_metrics(score, &metrics);
            }
        }
        spawn_snake(commands, segments_res, rng);
    }
}

fn snake_eating(
    mut food_timer: ResMut<FoodTimer>,
    mut clear_color: ResMut<ClearColor>,
    mut commands: Commands,
    mut growth_writer: EventWriter<GrowthEvent>,
    food_positions: Query<(Entity, &Position), With<Food>>,
    head_positions: Query<(&Position, &Player), With<SnakeHead>>,
) {
    for (head_pos, &player) in head_positions.iter() {
        for (ent, food_pos) in food_positions.iter() {
            if food_pos == head_pos {
                growth_writer.send(GrowthEvent(player));
                clear_color.0 = BACKGROUND_COLOR;
                commands.entity(ent).despawn();
                food_timer.0 = Some(4);
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
        if segments[player.index()].len() == 10 {
            game_over_writer.send(GameOverEvent(
                Some(player),
                GameOverReason::MaxLengthReached,
            ));
        } else {
            segments[player.index()].push(spawn_segment(
                &mut commands,
                last_tail_position.0[player.index()].unwrap(),
                player,
            ));
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

fn food_spawner(
    mut commands: Commands,
    mut food_timer: ResMut<FoodTimer>,
    pause: Res<Pause>,
    mut rng: ResMut<SmallRng>,
    segments: Query<(&SnakeSegment, &Position)>,
) {
    if pause.0 > 0 {
        return;
    }

    if let Some(time) = &mut food_timer.0 {
        if *time == 0 {
            let pos = loop {
                let x = rng.gen_range(0..ARENA_WIDTH) as i32;
                let y = rng.gen_range(0..ARENA_HEIGHT) as i32;
                let pos = Position { x, y };
                if !segments.iter().any(|(_, p)| *p == pos) {
                    break pos;
                }
            };
            commands
                .spawn_bundle(SpriteBundle {
                    sprite: Sprite {
                        color: FOOD_COLOR,
                        ..default()
                    },
                    ..default()
                })
                .insert(Food)
                .insert(pos)
                .insert(Size::square(0.8));
            food_timer.0 = None;
        } else {
            *time -= 1;
        }
    }
}

pub fn base_app(app: &mut App, seed: u64, timstep: Option<f64>) -> &mut App {
    let mut main_system = SystemSet::new()
        .with_system(snake_movement_agent)
        .with_system(snake_movement.after(snake_movement_agent))
        .with_system(snake_eating.after(snake_movement))
        .with_system(snake_growth.after(snake_eating))
        .with_system(food_spawner.after(snake_growth));
    if let Some(timstep) = timstep {
        main_system = main_system.with_run_criteria(FixedTimestep::step(timstep));
    }
    app.insert_resource(ClearColor(Color::rgb(0.04, 0.04, 0.04)))
        .insert_resource(Pause(0))
        .insert_resource(FoodTimer(Some(4)))
        .add_startup_system(setup_camera)
        .add_startup_system(spawn_snake)
        .insert_resource(SmallRng::seed_from_u64(seed))
        .insert_resource(SnakeSegments::default())
        .insert_resource(LastTailPosition::default())
        .add_event::<GrowthEvent>()
        .add_event::<GameOverEvent>()
        .add_system_set(main_system)
        .add_system(game_over.after(snake_movement))
}

pub fn run(agent_path: Option<String>, agent2_path: Option<String>, easy_mode: bool) {
    let opponent: Box<dyn Agent> = match agent_path {
        Some(path) => Box::new(RogueNetAgent::load(&path)),
        None => Box::new(RandomAgent::from_seed(1)),
    };
    let player: Option<Box<dyn Agent>> = match agent2_path {
        Some(path) => Some(Box::new(RogueNetAgent::load(&path))),
        None => None,
    };
    base_app(&mut App::new(), 0, Some(0.150))
        .insert_resource(WindowDescriptor {
            title: "Snake!".to_string(),
            width: 500.0,
            height: 500.0,
            ..default()
        })
        .insert_non_send_resource(Players([player, Some(opponent)]))
        .insert_resource(Config { easy_mode })
        .add_system(snake_movement_input.before(snake_movement))
        .add_system_set_to_stage(
            CoreStage::PostUpdate,
            SystemSet::new()
                .with_system(position_translation)
                .with_system(size_scaling),
        )
        .add_plugins(DefaultPlugins)
        .run();
}

pub fn run_headless(agents: [Box<dyn Agent>; 2], seed: u64) {
    let [a1, a2] = agents;
    base_app(&mut App::new(), seed, None)
        .insert_resource(ScheduleRunnerSettings::run_loop(Duration::from_secs_f64(
            0.0,
        )))
        .insert_resource(Config { easy_mode: false })
        .insert_non_send_resource(Players([Some(a1), Some(a2)]))
        .add_plugins(MinimalPlugins)
        .run();
}
