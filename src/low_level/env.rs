use rustc_hash::FxHashMap;

pub type EntityId = u64;
pub type ActionType = String;
pub type EntityType = String;

// Could have a `SingleAgentEnv` that implement `Environment`
// Could maybe use const generic for `agents`, return fixed size slices?
pub trait Environment {
    fn obs_space(&self) -> ObsSpace;
    fn action_space(&self) -> Vec<(ActionType, ActionSpace)>;
    fn agents() -> usize;

    #[allow(clippy::vec_box)]
    fn reset(&mut self) -> Vec<Box<Observation>>;
    #[allow(clippy::vec_box)]
    fn act(&mut self, action: &[Vec<Option<Action>>]) -> Vec<Box<Observation>>;
    fn close(&mut self) {}
}

#[derive(Debug, Clone)]
pub enum ActionSpace {
    Categorical { choices: Vec<String> },
    SelectEntity,
}

#[derive(Debug, Clone)]
pub enum ActionMask {
    DenseCategorical {
        actors: Vec<EntityId>,
        mask: Option<Vec<bool>>,
    },
    SelectEntity {
        actors: Vec<EntityId>,
        actees: Vec<EntityId>,
    },
}

#[derive(Debug, Clone)]
pub struct ObsSpace {
    pub entities: Vec<(EntityType, Entity)>,
}

#[derive(Debug, Clone)]
pub enum Action {
    Categorical {
        actors: Vec<EntityId>,
        action: Vec<usize>,
    },
    SelectEntity {
        actors: Vec<EntityId>,
        actees: Vec<EntityId>,
    },
}

#[derive(Debug, Clone)]
pub struct Entity {
    pub features: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CompactFeatures {
    pub counts: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug, Clone)]
pub struct Observation {
    pub features: CompactFeatures,
    // Maps each player to (optional) list of IDs for all entities
    pub ids: Vec<Option<Vec<EntityId>>>,
    pub actions: Vec<Option<ActionMask>>,

    pub done: bool,
    pub reward: f32,
    pub metrics: FxHashMap<String, f32>,
}
