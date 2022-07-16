use std::boxed::Box;
use std::collections::VecDeque;

use rand::prelude::*;
use rustc_hash::FxHashMap;

use crate::low_level::{
    Action, ActionMask, ActionSpace, ActionType, CompactFeatures, Entity, Environment, ObsSpace,
    Observation,
};

#[derive(Debug)]
pub struct MultiSnake {
    board_size: usize,
    num_snakes: usize,
    max_snake_length: usize,
    max_step: usize,

    rng: SmallRng,
    snakes: Vec<Snake>,
    food: Vec<Food>,
    game_over: bool,
    last_score: f32,
    score: f32,
    step: usize,
}

#[derive(Debug)]
struct Food {
    x: i32,
    y: i32,
    color: usize,
}

#[derive(Debug)]
struct Snake {
    color: usize,
    segments: VecDeque<(i32, i32)>,
}

impl MultiSnake {
    pub fn new(
        board_size: usize,
        num_snakes: usize,
        max_snake_length: usize,
        max_step: usize,
        seed: u64,
    ) -> MultiSnake {
        MultiSnake {
            board_size,
            num_snakes,
            max_snake_length,
            max_step,

            rng: SmallRng::seed_from_u64(seed),
            snakes: Vec::with_capacity(num_snakes),
            food: Vec::with_capacity(num_snakes),
            game_over: false,
            last_score: 0.0,
            score: 0.0,
            step: 0,
        }
    }

    fn observe(&self) -> Box<Observation> {
        let nhead = self.snakes.len();
        let nsegment = self
            .snakes
            .iter()
            .map(|s| s.segments.len() - 1)
            .sum::<usize>();
        let nfood = self.food.len();

        let mut features = CompactFeatures {
            counts: vec![nhead, nsegment, nfood],
            data: Vec::with_capacity(4 * nhead + 3 * nsegment + 3 * nfood),
        };
        for snake in &self.snakes {
            let (x, y) = *snake.segments.front().unwrap();
            features.data.push(x as f32);
            features.data.push(y as f32);
            features.data.push(snake.color as f32);
            features.data.push(self.step as f32);
        }
        for snake in &self.snakes {
            for &(x, y) in snake.segments.iter().skip(1) {
                features.data.push(x as f32);
                features.data.push(y as f32);
                features.data.push(snake.color as f32);
            }
        }
        for f in &self.food {
            features.data.push(f.x as f32);
            features.data.push(f.y as f32);
            features.data.push(f.color as f32);
        }

        let actions = vec![Some(ActionMask::DenseCategorical {
            actors: (0..self.num_snakes as u64).collect(),
            mask: None,
        })];
        let ids = vec![Some((0..self.num_snakes as u64).collect()), None, None];

        Box::new(Observation {
            features,
            actions,
            done: self.game_over,
            reward: self.score - self.last_score,
            ids,
            metrics: FxHashMap::default(),
        })
    }

    fn random_empty_tile(&mut self) -> (i32, i32) {
        for _ in 0..10000 {
            let x = self.rng.gen_range(0..self.board_size as i32);
            let y = self.rng.gen_range(0..self.board_size as i32);
            if self.snakes.iter().any(|s| s.segments.contains(&(x, y))) {
                continue;
            }
            if self.food.iter().any(|f| f.x == x && f.y == y) {
                continue;
            }
            return (x, y);
        }
        panic!("No empty tile found. {:?}", self);
    }
}

impl Environment for MultiSnake {
    fn obs_space(&self) -> ObsSpace {
        ObsSpace {
            entities: vec![
                (
                    "SnakeHead".to_string(),
                    Entity {
                        features: vec![
                            "x".to_string(),
                            "y".to_string(),
                            "color".to_string(),
                            "step".to_string(),
                        ],
                    },
                ),
                (
                    "SnakeBody".to_string(),
                    Entity {
                        features: vec!["x".to_string(), "y".to_string(), "color".to_string()],
                    },
                ),
                (
                    "Food".to_string(),
                    Entity {
                        features: vec!["x".to_string(), "y".to_string(), "color".to_string()],
                    },
                ),
            ],
        }
    }

    fn action_space(&self) -> Vec<(ActionType, ActionSpace)> {
        vec![(
            "move".to_string(),
            ActionSpace::Categorical {
                choices: vec![
                    "up".to_string(),
                    "down".to_string(),
                    "left".to_string(),
                    "right".to_string(),
                ],
            },
        )]
    }

    fn reset(&mut self) -> Vec<Box<Observation>> {
        self.snakes.clear();
        self.food.clear();
        self.game_over = false;
        self.last_score = 0.0;
        self.score = 0.0;
        self.step = 0;
        for color in 0..self.num_snakes {
            let mut snake = Snake {
                color,
                segments: VecDeque::new(),
            };
            snake.segments.push_front(self.random_empty_tile());
            self.snakes.push(snake);
            let (x, y) = self.random_empty_tile();
            self.food.push(Food { x, y, color });
        }
        vec![self.observe()]
    }

    fn act(&mut self, action: &[Vec<Option<Action>>]) -> Vec<Box<Observation>> {
        self.last_score = self.score;
        self.step += 1;
        let mut food_to_spawn = vec![];
        // Execute all actions
        if let Some(Action::Categorical { actors, action }) = &action[0][0] {
            for (id, choice) in actors.iter().zip(action) {
                // Calculate new head position
                let (mut x, mut y) = self.snakes[*id as usize].segments.front().unwrap();
                match choice {
                    0 => y += 1,
                    1 => y -= 1,
                    2 => x -= 1,
                    3 => x += 1,
                    _ => panic!("Invalid action '{choice}'"),
                }
                // Check for collision with board boundaries
                if x < 0 || x >= self.board_size as i32 || y < 0 || y >= self.board_size as i32 {
                    self.game_over = true;
                }
                // Check for collision with self or other snakes
                if self.snakes.iter().any(|s| s.segments.contains(&(x, y))) {
                    self.game_over = true;
                }
                // Check for collision with food
                let mut ate_food = false;
                for (i, food) in self.food.iter_mut().enumerate() {
                    if food.x == x && food.y == y {
                        if food.color != self.snakes[*id as usize].color {
                            self.game_over = true;
                        } else if self.snakes[*id as usize].segments.len() < self.max_snake_length {
                            self.score +=
                                1.0 / ((self.max_snake_length - 1) as f32) / self.num_snakes as f32;
                            ate_food = true;
                        }
                        food_to_spawn.push(i);
                        break;
                    }
                }
                // Place new head
                self.snakes[*id as usize].segments.push_front((x, y));
                // Remove tail unless we ate food and we are not at the max length
                if !ate_food {
                    self.snakes[*id as usize].segments.pop_back();
                }
            }
        }
        // Spawn new food
        for i in food_to_spawn {
            let (x, y) = self.random_empty_tile();
            self.food[i].x = x;
            self.food[i].y = y;
        }
        // Check for game over
        if self.step >= self.max_step {
            self.game_over = true;
        }
        if self
            .snakes
            .iter()
            .all(|s| s.segments.len() == self.max_snake_length)
        {
            self.game_over = true;
        }

        vec![self.observe()]
    }

    fn agents() -> usize {
        1
    }
}
