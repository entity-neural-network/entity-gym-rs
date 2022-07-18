use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::low_level::{
    Action, ActionMask, ActionSpace, ActionType, CompactFeatures, Entity, Environment, ObsSpace,
    Observation,
};
use crossbeam::channel::{bounded, Receiver, Sender};

use super::{ActionReceiver, Agent, Featurizable, InnerActionReceiver, Obs};

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

    // The `obs_remaining` counters are shared between all agents that connect to the same
    // environment and is used to detect deadlocks that would be caused by an agent awaiting
    // an action before all agents have received an observation.
    // When acting, the agent decrements the counter. When receiving an observation, the agent
    // increments the counter. To prevent an agent from observing a counter reset before it has
    // reached the next step, we use two counters which we swap every step.
    // The `iremaining` variable keeps track of the counter that is currently in use.
    obs_remaining: [Arc<AtomicUsize>; 2],
    iremaining: usize,
    observation_sent: bool,
    agent_count: usize,
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

    fn agents(&self) -> usize {
        self.action.len()
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
    fn act_raw(&mut self, action: &str, _num_actions: u64, obs: &Obs) -> Option<u64> {
        self.send_obs_raw(action, obs);
        let remaining = self.obs_remaining[self.iremaining].load(Ordering::SeqCst);
        if remaining != 0 && (remaining != self.agent_count || self.agent_count == 1) {
            panic!("TrainAgent::act called before all agents have received observations. This is not allowed. If you have multiple agents, call the `act_async` on every agent before awaiting any actions.");
        }
        match self.action.recv() {
            Ok(action) => {
                self.obs_remaining[self.iremaining].store(self.agent_count, Ordering::SeqCst);
                self.iremaining = 1 - self.iremaining;
                self.observation_sent = false;
                Some(action)
            }
            Err(_) => None,
        }
    }

    fn act_async_raw(&mut self, action: &str, _num_actions: u64, obs: &Obs) -> ActionReceiver<u64> {
        self.send_obs_raw(action, obs);
        self.observation_sent = false;
        self.iremaining = 1 - self.iremaining;
        ActionReceiver {
            inner: InnerActionReceiver::Receiver {
                receiver: self.action.clone(),
                observations_remaining: self.obs_remaining[1 - self.iremaining].clone(),
                agent_count: self.agent_count,
                phantom: Default::default(),
            },
        }
    }

    fn game_over(&mut self, score: f32) {
        let obs = Observation {
            features: CompactFeatures {
                counts: vec![0; self.entity_names.len()],
                data: vec![],
            },
            ids: vec![None],
            actions: vec![None],
            done: true,
            reward: score - self.score.unwrap_or(0.0),
            metrics: Default::default(),
        };
        self.score = None;
        let _ = self.observation.send(obs);
    }
}

impl TrainAgent {
    fn send_obs_raw(&mut self, _action: &str, obs: &Obs) {
        assert!(
            self.observation_sent == false,
            "Observation already sent, await the next action before sending a new observation."
        );
        self.obs_remaining[self.iremaining].fetch_sub(1, Ordering::SeqCst);

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
                mask: None,
            })],
            done: obs.done,
            reward: obs.score - last_score,
            metrics: Default::default(),
        };
        let _ = self.observation.send(observation);
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
        let obs_remaining = [
            Arc::new(AtomicUsize::new(num_agents)),
            Arc::new(AtomicUsize::new(num_agents)),
        ];
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
                obs_remaining: obs_remaining.clone(),
                iremaining: 0,
                observation_sent: false,
                agent_count: num_agents,
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
                obs_remaining: [Arc::new(AtomicUsize::new(1)), Arc::new(AtomicUsize::new(1))],
                iremaining: 0,
                observation_sent: false,
                agent_count: 1,
            },
        )
    }
}
