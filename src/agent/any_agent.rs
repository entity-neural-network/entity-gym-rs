use super::{Action, ActionReceiver, Agent, Obs, RandomAgent, RogueNetAgent, TrainAgent};

pub enum AnyAgent {
    Random(RandomAgent),
    TrainAgent(TrainAgent),
    RogueNetAgent(Box<RogueNetAgent>),
}

impl AnyAgent {
    pub fn random() -> AnyAgent {
        AnyAgent::Random(RandomAgent::default())
    }

    pub fn random_seeded(seed: u64) -> AnyAgent {
        AnyAgent::Random(RandomAgent::new(seed))
    }

    pub fn rogue_net(path: &str) -> AnyAgent {
        AnyAgent::RogueNetAgent(Box::new(RogueNetAgent::load(path)))
    }

    pub fn train(agent: TrainAgent) -> AnyAgent {
        AnyAgent::TrainAgent(agent)
    }
}

impl Agent for AnyAgent {
    fn act<A: Action>(&mut self, obs: &Obs) -> Option<A> {
        match self {
            AnyAgent::Random(agent) => agent.act::<A>(obs),
            AnyAgent::TrainAgent(agent) => agent.act::<A>(obs),
            AnyAgent::RogueNetAgent(agent) => agent.act::<A>(obs),
        }
    }

    fn act_async<A: Action>(&mut self, obs: &Obs) -> ActionReceiver<A> {
        match self {
            AnyAgent::Random(agent) => agent.act_async::<A>(obs),
            AnyAgent::TrainAgent(agent) => agent.act_async::<A>(obs),
            AnyAgent::RogueNetAgent(agent) => agent.act_async::<A>(obs),
        }
    }

    fn game_over(&mut self, score: f32) {
        match self {
            AnyAgent::Random(agent) => agent.game_over(score),
            AnyAgent::TrainAgent(agent) => agent.game_over(score),
            AnyAgent::RogueNetAgent(agent) => agent.game_over(score),
        }
    }
}
