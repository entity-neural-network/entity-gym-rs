use std::sync::{Arc, Mutex};

use super::{Action, Obs, RandomAgent, RogueNetAgent, TrainAgent};

pub trait Agent {
    fn act<A: Action>(&mut self, obs: Obs) -> Option<A>;

    fn game_over(&mut self) {}
}

pub enum AnyAgent {
    Random(RandomAgent),
    // TODO: get rid of mutex?
    TrainAgent(Arc<Mutex<TrainAgent>>),
    RogueNetAgent(Box<RogueNetAgent>),
}

impl AnyAgent {
    pub fn random() -> AnyAgent {
        AnyAgent::Random(RandomAgent::default())
    }

    pub fn rogue_net(path: &str) -> AnyAgent {
        AnyAgent::RogueNetAgent(Box::new(RogueNetAgent::load(path)))
    }

    pub fn train(agent: TrainAgent) -> AnyAgent {
        AnyAgent::TrainAgent(Arc::new(Mutex::new(agent)))
    }
}

impl Agent for AnyAgent {
    fn act<A: Action>(&mut self, obs: Obs) -> Option<A> {
        match self {
            AnyAgent::Random(agent) => agent.act::<A>(obs),
            AnyAgent::TrainAgent(agent) => agent.lock().unwrap().act::<A>(obs),
            AnyAgent::RogueNetAgent(agent) => agent.act::<A>(obs),
        }
    }

    fn game_over(&mut self) {
        match self {
            AnyAgent::Random(agent) => agent.game_over(),
            AnyAgent::TrainAgent(agent) => agent.lock().unwrap().game_over(),
            AnyAgent::RogueNetAgent(agent) => agent.game_over(),
        }
    }
}
