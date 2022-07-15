use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use atomicbox::AtomicOptionBox;
use crossbeam::channel::{bounded, Receiver, Sender};
use crossbeam::sync::{Parker, Unparker};
use ragged_buffer::ragged_buffer::RaggedBuffer;
use std::thread;

use super::{Action, ActionMask, ActionSpace, ActionType, Environment, ObsSpace, Observation};

pub struct VecEnv {
    inner: Arc<VecEnvInner>,
    tasks: Vec<Sender<Task>>,
    wait_on_obs: Parker,

    pub num_feats: Vec<usize>,
    pub obs_space: ObsSpace,
    pub action_space: Vec<(ActionType, ActionSpace)>,
    pub num_envs: usize,

    total_send: u64,
    total_wait: u64,
    total_collect: u64,
}

enum Task {
    Exit,
    Reset,
    RawBatchAct(Arc<Vec<Option<RaggedBuffer<i64>>>>),
}

struct VecEnvInner {
    obs: Vec<AtomicOptionBox<Observation>>,
    completed: AtomicUsize,
    wake_obs: Unparker,
}

impl VecEnv {
    #[allow(clippy::mutex_atomic)]
    pub fn new<T: Environment + Send + 'static>(
        create_env: Arc<dyn Fn(u64) -> T + Send + Sync>,
        num_envs: usize,
        threads: usize,
        first_env_index: u64,
    ) -> VecEnv {
        let parker = Parker::new();
        let unparker = parker.unparker().clone();
        let inner = Arc::new(VecEnvInner {
            obs: (0..num_envs).map(|_| AtomicOptionBox::none()).collect(),
            completed: AtomicUsize::new(0),
            wake_obs: unparker,
        });
        let mut senders = Vec::new();
        for i in 0..threads {
            let (task_tx, task_rx) = bounded(num_envs);
            let inner = inner.clone();
            let create_env = create_env.clone();
            thread::spawn(move || {
                inner.worker(task_rx, create_env, i, threads, num_envs, first_env_index)
            });
            senders.push(task_tx)
        }
        let env = create_env(99999);
        VecEnv {
            inner,
            tasks: senders,
            wait_on_obs: parker,

            num_feats: env
                .obs_space()
                .entities
                .iter()
                .map(|(_, e)| e.features.len())
                .collect(),
            obs_space: env.obs_space(),
            action_space: env.action_space(),
            num_envs,

            total_send: 0,
            total_wait: 0,
            total_collect: 0,
        }
    }

    pub fn reset(&mut self) -> Vec<Box<Observation>> {
        self.inner.completed.store(0, Ordering::SeqCst);
        for task in &mut self.tasks {
            task.send(Task::Reset).unwrap();
        }
        self.wait_on_obs.park();
        self.inner
            .obs
            .iter()
            .map(|obs| obs.take(Ordering::SeqCst).unwrap())
            .collect()
    }

    pub fn act(&mut self, actions: Vec<Option<RaggedBuffer<i64>>>) -> Vec<Box<Observation>> {
        //println!();
        let start_time = std::time::Instant::now();
        self.inner.completed.store(0, Ordering::SeqCst);
        let actions = Arc::new(actions);
        for task in &self.tasks {
            task.send(Task::RawBatchAct(actions.clone())).unwrap();
        }
        let send_ns = start_time.elapsed().as_nanos();
        //println!("Sending actions: {} ns", send_ns);
        drop(actions);
        self.total_send += send_ns as u64;
        let start_time = std::time::Instant::now();
        self.wait_on_obs.park();
        let wait_ns = start_time.elapsed().as_nanos();
        //println!("Await obs:       {} ns", wait_ns);
        self.total_wait += wait_ns as u64;
        let start_time = std::time::Instant::now();
        let obss = self
            .inner
            .obs
            .iter()
            .map(|obs| obs.take(Ordering::SeqCst).unwrap())
            .collect();
        let collect_ns = start_time.elapsed().as_nanos();
        //println!("Collecting obs:  {} ns", collect_ns);
        self.total_collect += collect_ns as u64;
        obss
    }
}

impl VecEnvInner {
    fn worker<T: Environment>(
        &self,
        rx: Receiver<Task>,
        create_env: Arc<dyn Fn(u64) -> T>,
        thread_id: usize,
        nthread: usize,
        total_envs: usize,
        env_offset: u64,
    ) {
        let local_envs = total_envs / nthread
            + if thread_id < total_envs % nthread {
                1
            } else {
                0
            };
        assert!(local_envs % T::agents() == 0);
        let env_offset =
            thread_id * (total_envs / nthread) + total_envs % nthread + env_offset as usize;
        let mut envs = (0..local_envs / T::agents())
            .map(|i| create_env((i + env_offset).try_into().unwrap()))
            .collect::<Vec<_>>();
        let mut action_masks = vec![];
        loop {
            let task = rx.recv().unwrap();
            match task {
                Task::Exit => break,
                Task::Reset => {
                    action_masks.clear();
                    for (i, env) in envs.iter_mut().enumerate() {
                        let env_id = i * T::agents() + env_offset;
                        let obs = env.reset();
                        for (j, obs) in obs.into_iter().enumerate() {
                            action_masks.push(obs.actions.clone());
                            self.obs[env_id + j].store(Some(obs), Ordering::SeqCst);
                        }
                    }
                    if self
                        .completed
                        .fetch_add(envs.len() * T::agents(), Ordering::SeqCst)
                        == total_envs - envs.len() * T::agents()
                    {
                        self.wake_obs.unpark();
                    }
                }
                Task::RawBatchAct(ragged_actions) => {
                    let mut new_action_masks = Vec::with_capacity(action_masks.len());
                    for (i, env) in envs.iter_mut().enumerate() {
                        let mut actions = Vec::with_capacity(T::agents());
                        for agent in 0..T::agents() {
                            let env_id = env_offset + i * T::agents() + agent;
                            let action = ragged_actions
                                .iter()
                                .enumerate()
                                .map(|(idx_act_type, a)| match a {
                                    Some(a) => {
                                        let subarray = a.subarrays[env_id].clone();
                                        match &action_masks[i * T::agents() + agent][idx_act_type] {
                                            Some(ActionMask::DenseCategorical {
                                                actors, ..
                                            }) => Some(Action::Categorical {
                                                actors: actors.clone(),
                                                action: a.data[subarray]
                                                    .iter()
                                                    .map(|x| *x as usize)
                                                    .collect(),
                                            }),
                                            Some(ActionMask::SelectEntity { actors, actees }) => {
                                                Some(Action::SelectEntity {
                                                    actors: actors.clone(),
                                                    actees: a.data[subarray]
                                                        .iter()
                                                        .map(|x| actees[*x as usize])
                                                        .collect(),
                                                })
                                            }
                                            None => None,
                                        }
                                    }
                                    None => None,
                                })
                                .collect::<Vec<_>>();
                            actions.push(action);
                        }
                        let mut obs = env.act(&actions);
                        if obs[0].done {
                            let mut onew = env.reset();
                            for i in 0..obs.len() {
                                onew[i].reward = obs[i].reward;
                                onew[i].done = obs[i].done;
                            }
                            obs = onew;
                        }
                        let env_id = env_offset + i * T::agents();
                        for (j, obs) in obs.into_iter().enumerate() {
                            new_action_masks.push(obs.actions.clone());
                            self.obs[env_id + j].store(Some(obs), Ordering::SeqCst);
                        }
                    }
                    if self
                        .completed
                        .fetch_add(envs.len() * T::agents(), Ordering::SeqCst)
                        == total_envs - envs.len() * T::agents()
                    {
                        self.wake_obs.unpark();
                    }
                    action_masks = new_action_masks;
                }
            }
        }
    }
}

impl Drop for VecEnv {
    fn drop(&mut self) {
        println!("Total send: {} ms", self.total_send / 1_000_000);
        println!("Total wait: {} ms", self.total_wait / 1_000_000);
        println!("Total collect: {} ms", self.total_collect / 1_000_000);
        // TODO: bounded channel, there could still be tasks in the queue
        // TODO: also only need to send num_threads exits
        for tx in &self.tasks {
            tx.send(Task::Exit).unwrap();
        }
    }
}
