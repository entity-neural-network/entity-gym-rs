#![allow(clippy::too_many_arguments)]
mod ai;
#[cfg(feature = "python")]
pub mod python;

use std::time::Duration;

use ai::{snake_movement_agent, Opponents, Players};
use bevy::app::ScheduleRunnerSettings;
use bevy::core::FixedTimestep;
use bevy::prelude::*;
use entity_gym_rs::agent::{Action, Agent, RandomAgent, RogueNetAgent};
use rand::prelude::SmallRng;
use rand::{Rng, SeedableRng};

const HEAD_COLOR: [Color; 2] = [Color::rgb(0.6, 0.6, 1.0), Color::rgb(1.0, 0.6, 0.6)];
const FOOD_COLOR: Color = Color::rgb(1.0, 0.0, 1.0);
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
struct ZPos {
    z: i32,
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
    last_direction: Direction,
}

struct GameOverEvent(Option<Player>, GameOverReason);

enum GameOverReason {
    HeadCollision,
    MaxLengthReached,
    SnakeCollision,
    WallCollision,
}

struct ResetGameEvent;

#[derive(Component)]
struct Level {
    level: usize,
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

fn spawn_level_text(mut commands: Commands, asset_server: Res<AssetServer>) {
    let font = asset_server.load("fonts/FiraSans-Bold.ttf");
    let text_style = TextStyle {
        font,
        font_size: 60.0,
        color: Color::DARK_GRAY,
    };
    let text_alignment = TextAlignment {
        vertical: VerticalAlign::Center,
        horizontal: HorizontalAlign::Center,
    };
    commands
        .spawn_bundle(Text2dBundle {
            text: Text::with_section("level 1", text_style, text_alignment),
            transform: Transform::from_translation(Vec3::new(0.0, 00.0, -0.0)),
            ..default()
        })
        .insert(Level { level: 1 });
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
            let y = rng.gen_range(1..(ARENA_HEIGHT - 4)) as i32;
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
                        last_direction: Direction::Up,
                    })
                    .insert(SnakeSegment)
                    .insert(Position { x, y })
                    .insert(ZPos { z: 0 })
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
        .insert(ZPos { z: 1 })
        .insert(Size::square(0.65))
        .insert(player)
        .id()
}

#[allow(clippy::too_many_arguments)]
fn snake_movement(
    config: Res<Config>,
    mut last_tail_position: ResMut<LastTailPosition>,
    mut game_over_writer: EventWriter<GameOverEvent>,
    mut reset_game_writer: EventWriter<ResetGameEvent>,
    mut pause: ResMut<Pause>,
    segments: ResMut<SnakeSegments>,
    mut heads: Query<(Entity, &mut SnakeHead, &Player)>,
    mut positions: Query<&mut Position>,
) {
    if pause.0 > 0 {
        pause.0 -= 1;
        if pause.0 == 1 {
            reset_game_writer.send(ResetGameEvent);
        }
        return;
    }
    #[allow(clippy::needless_collect)]
    let all_segment_positions = segments
        .iter()
        // Drop final segment since it will move to next position
        .flat_map(|s| s[0..s.len() - 1].iter())
        .map(|e| *positions.get_mut(*e).unwrap())
        .collect::<Vec<Position>>();
    let mut head_positions = vec![];
    for (head_entity, mut head, p) in heads.iter_mut() {
        let segment_positions = segments[p.index()]
            .iter()
            .map(|e| *positions.get_mut(*e).unwrap())
            .collect::<Vec<Position>>();
        let mut head_pos = positions.get_mut(head_entity).unwrap();
        head.last_direction = head.direction;
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
        if dir != head.last_direction.opposite() {
            head.direction = dir;
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn game_over(
    mut reader: EventReader<GameOverEvent>,
    mut players: NonSendMut<Players>,
    opponents: Res<Opponents>,
    mut pause: ResMut<Pause>,
    mut segments: Query<(&mut Sprite, &Player, &SnakeSegment), Without<SnakeHead>>,
    mut level: Query<&mut Level>,
    segments_res: ResMut<SnakeSegments>,
) {
    if let Some(GameOverEvent(winner, reason)) = reader.iter().next() {
        let game_over_reason_str = match reason {
            GameOverReason::WallCollision => "game_over/wall_collision",
            GameOverReason::SnakeCollision => "game_over/snake_collision",
            GameOverReason::HeadCollision => "game_over/head_collision",
            GameOverReason::MaxLengthReached => "game_over/max_length_reached",
        };
        pause.0 = 10;

        if matches!(reason, GameOverReason::MaxLengthReached) {
            // If player wins by reaching the max length, set color of all segments to the winner's color.
            for (mut sprite, player, _) in segments.iter_mut() {
                if winner.iter().any(|p| p == player) {
                    sprite.color = HEAD_COLOR[player.index()];
                }
            }
        }

        if let Some(mut level) = level.iter_mut().next() {
            match winner {
                Some(Player::Blue) if level.level < opponents.0.len() => level.level += 1,
                Some(Player::Red) if level.level > 1 => level.level -= 1,
                _ => {}
            }
        }

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
    }
}

fn reset_game(
    mut reader: EventReader<ResetGameEvent>,
    mut commands: Commands,
    segments: Query<Entity, With<SnakeSegment>>,
    segments_res: ResMut<SnakeSegments>,
    mut food_timer: ResMut<FoodTimer>,
    mut level_text: Query<(&Level, &mut Text)>,
    rng: ResMut<SmallRng>,
    food: Query<Entity, With<Food>>,
) {
    if let Some(ResetGameEvent) = reader.iter().next() {
        for ent in food.iter().chain(segments.iter()) {
            commands.entity(ent).despawn();
        }
        food_timer.0 = Some(4);
        spawn_snake(commands, segments_res, rng);
        if let Some((level, mut text)) = level_text.iter_mut().next() {
            text.sections[0].value = format!("level {}", level.level);
        }
    }
}

fn snake_eating(
    mut food_timer: ResMut<FoodTimer>,
    mut commands: Commands,
    mut growth_writer: EventWriter<GrowthEvent>,
    food_positions: Query<(Entity, &Position), With<Food>>,
    head_positions: Query<(&Position, &Player), With<SnakeHead>>,
) {
    for (head_pos, &player) in head_positions.iter() {
        for (ent, food_pos) in food_positions.iter() {
            if food_pos == head_pos {
                growth_writer.send(GrowthEvent(player));
                //clear_color.0 = BACKGROUND_COLOR;
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
            2.0,
        );
    }
}

fn position_translation(windows: Res<Windows>, mut q: Query<(&Position, &ZPos, &mut Transform)>) {
    fn convert(pos: f32, bound_window: f32, bound_game: f32) -> f32 {
        let tile_size = bound_window / bound_game;
        pos / bound_game * bound_window - (bound_window / 2.) + (tile_size / 2.)
    }
    let window = windows.get_primary().unwrap();
    for (pos, zpos, mut transform) in q.iter_mut() {
        transform.translation = Vec3::new(
            convert(pos.x as f32, window.width() as f32, ARENA_WIDTH as f32),
            convert(pos.y as f32, window.height() as f32, ARENA_HEIGHT as f32),
            zpos.z as f32,
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
                if !segments.iter().any(|(_, p)| p.x == x && p.y == y) {
                    break Position { x, y };
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
                .insert(ZPos { z: 0 })
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
        .add_event::<ResetGameEvent>()
        .add_system_set(main_system)
        .add_system(game_over.after(snake_movement))
        .add_system(reset_game.after(snake_movement))
}

pub fn run(
    agent_path: Option<String>,
    agent2_path: Option<String>,
    opponents: Vec<String>,
    easy_mode: bool,
) {
    let opponent: Box<dyn Agent> = match agent_path {
        Some(path) => Box::new(RogueNetAgent::load(&path)),
        None => Box::new(RandomAgent::from_seed(1)),
    };
    let player: Option<Box<dyn Agent>> = match agent2_path {
        Some(path) => Some(Box::new(RogueNetAgent::load(&path))),
        None => None,
    };
    let opponents = opponents
        .into_iter()
        .map(|path| RogueNetAgent::load(&path))
        .collect::<Vec<_>>();
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
        .add_startup_system(spawn_level_text)
        .insert_resource(Opponents(opponents))
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
        .insert_resource(Opponents(vec![]))
        .insert_non_send_resource(Players([Some(a1), Some(a2)]))
        .add_plugins(MinimalPlugins)
        .run();
}
