use crate::low_level::{
    Action, ActionMask, ActionSpace, ActionType, CompactFeatures, Entity, Environment, ObsSpace,
    Observation,
};
use crossbeam::channel::{bounded, Receiver, Sender};

use super::{Agent, Featurizable, Obs};

pub struct TrainAgentEnv {
    obs_space: ObsSpace,
    action_space: Vec<(String, ActionSpace)>,
    action: Vec<Sender<u64>>,
    observation: Vec<Receiver<Observation>>,
}

pub struct TrainAgent {
    action: Receiver<u64>,
    observation: Sender<Observation>,
    entity_names: Vec<String>,
    score: Option<f32>,
}

#[derive(Debug, Clone, Default)]
pub struct TrainEnvBuilder {
    entities: Vec<(String, Entity)>,
    actions: Vec<(String, ActionSpace)>,
}

impl Environment for TrainAgentEnv {
    fn obs_space(&self) -> ObsSpace {
        self.obs_space.clone()
    }

    fn action_space(&self) -> Vec<(ActionType, ActionSpace)> {
        self.action_space.clone()
    }

    fn agents() -> usize {
        1
    }

    fn reset(&mut self) -> Vec<Box<Observation>> {
        self.observation
            .iter()
            .map(|obs| Box::new(obs.recv().unwrap()))
            .collect()
    }

    fn act(&mut self, action: &[Vec<Option<Action>>]) -> Vec<Box<Observation>> {
        assert!(action.len() == self.action.len());
        for (sender, action) in self.action.iter().zip(action.iter()) {
            assert!(action.len() == 1);
            match &action[0] {
                Some(Action::Categorical { actors: _, action }) => {
                    assert!(action.len() == 1);
                    sender.send(action[0] as u64).unwrap();
                }
                Some(_) => panic!("unexpected action"),
                None => {
                    panic!("No action provided");
                }
            }
        }
        self.observation
            .iter()
            .map(|obs| Box::new(obs.recv().unwrap()))
            .collect()
    }
}

impl Agent for TrainAgent {
    fn act<A: super::Action>(&mut self, obs: &Obs) -> Option<A> {
        let mut data = vec![];
        let mut counts = vec![];
        for name in &self.entity_names {
            match obs.entities.get(name.as_str()) {
                Some((feats, count, _)) => {
                    data.extend(feats.iter());
                    counts.push(*count);
                }
                None => {
                    counts.push(0);
                }
            }
        }
        // TODO: make noise when obs contains entity that is not in obs space
        let last_score = self.score.replace(obs.score).unwrap_or(obs.score);
        let observation = Observation {
            features: CompactFeatures { counts, data },
            ids: vec![None],
            actions: vec![Some(ActionMask::DenseCategorical {
                actors: vec![0],
                mask: Some(vec![true; A::num_actions() as usize]),
            })],
            done: obs.done,
            reward: obs.score - last_score,
            metrics: Default::default(),
        };
        self.observation.send(observation).ok()?;
        match self.action.recv() {
            Ok(action) => Some(A::from_u64(action)),
            Err(_) => None,
        }
    }

    fn game_over(&mut self) {
        let obs = Observation {
            features: CompactFeatures {
                counts: vec![0; self.entity_names.len()],
                data: vec![],
            },
            ids: vec![None],
            actions: vec![None],
            done: true,
            reward: 0.0,
            metrics: Default::default(),
        };
        self.score = None;
        let _ = self.observation.send(obs);
    }
}

impl TrainEnvBuilder {
    pub fn entity<E: Featurizable>(mut self) -> Self {
        assert!(
            self.entities.iter().all(|(n, _)| n != E::name()),
            "Already have an entity with name \"{}\"",
            E::name(),
        );
        self.entities.push((
            E::name().to_string(),
            Entity {
                features: E::feature_names().iter().map(|n| n.to_string()).collect(),
            },
        ));
        self
    }

    pub fn action<A: super::Action>(mut self) -> Self {
        assert!(
            self.actions.iter().all(|(n, _)| n != A::name()),
            "Already have an action with name \"{}\"",
            A::name(),
        );
        self.actions.push((
            A::name().to_string(),
            ActionSpace::Categorical {
                choices: A::labels().iter().map(|c| c.to_string()).collect(),
            },
        ));
        self
    }

    pub fn build_multiagent(self, num_agents: usize) -> (TrainAgentEnv, Vec<TrainAgent>) {
        let mut environment = TrainAgentEnv {
            obs_space: ObsSpace {
                entities: self.entities.to_vec(),
            },
            action_space: self.actions,
            action: vec![],
            observation: vec![],
        };
        let mut agents = vec![];

        for _ in 0..num_agents {
            let (action_tx, action_rx) = bounded(1);
            let (observation_tx, observation_rx) = bounded(1);
            let entity_names = self.entities.iter().map(|(n, _)| n.to_string()).collect();
            environment.action.push(action_tx);
            environment.observation.push(observation_rx);
            agents.push(TrainAgent {
                action: action_rx,
                observation: observation_tx,
                entity_names,
                score: None,
            });
        }

        (environment, agents)
    }

    pub fn build(self) -> (TrainAgentEnv, TrainAgent) {
        let (action_tx, action_rx) = bounded(1);
        let (observation_tx, observation_rx) = bounded(1);
        let entity_names = self.entities.iter().map(|(n, _)| n.to_string()).collect();
        (
            TrainAgentEnv {
                obs_space: ObsSpace {
                    entities: self.entities.into_iter().collect(),
                },
                action_space: self.actions,
                action: vec![action_tx],
                observation: vec![observation_rx],
            },
            TrainAgent {
                action: action_rx,
                observation: observation_tx,
                entity_names,
                score: None,
            },
        )
    }
}
