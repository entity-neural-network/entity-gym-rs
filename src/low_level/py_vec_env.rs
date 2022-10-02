use std::collections::hash_map::Entry;

use numpy::{PyArray1, PyReadonlyArrayDyn, ToPyArray};
use pyo3::prelude::*;
use ragged_buffer::monomorphs::{RaggedBufferBool, RaggedBufferF32, RaggedBufferI64};
use ragged_buffer::ragged_buffer::RaggedBuffer;
use rustc_hash::FxHashMap;

use super::{ActionMask, ActionSpace, Observation, VecEnv};

#[pyclass]
pub struct PyVecEnv {
    pub env: VecEnv,
}

#[pyclass]
pub struct VecObs {
    #[pyo3(get)]
    pub features: Vec<(String, RaggedBufferF32)>,
    #[allow(clippy::type_complexity)]
    #[pyo3(get)]
    pub action_masks: Vec<(
        String,
        (
            RaggedBufferI64,
            Option<RaggedBufferI64>,
            Option<RaggedBufferBool>,
        ),
    )>,
    #[pyo3(get)]
    pub reward: Py<PyArray1<f32>>,
    #[pyo3(get)]
    pub done: Py<PyArray1<bool>>,
    // (count, sum, max, min)
    #[pyo3(get)]
    pub metrics: FxHashMap<String, (usize, f32, f32, f32)>,
}

#[pymethods]
impl PyVecEnv {
    fn reset(&mut self, py: Python) -> VecObs {
        let obs = self.env.reset();
        self.merge_obs(py, &obs[..])
    }

    fn act(
        &mut self,
        py: Python,
        action: Vec<(PyReadonlyArrayDyn<i64>, PyReadonlyArrayDyn<i64>)>,
    ) -> VecObs {
        let obs = self.env.act(
            action
                .into_iter()
                .map(|(data, lengths)| {
                    let mut cumsum = 0usize;
                    let mut subarrays = Vec::with_capacity(lengths.len());
                    for len in lengths.iter().unwrap() {
                        let subarray = cumsum..(cumsum + *len as usize);
                        subarrays.push(subarray);
                        cumsum += *len as usize;
                    }
                    Some(RaggedBuffer::<i64> {
                        data: if data.is_empty() {
                            vec![]
                        } else {
                            data.iter().unwrap().copied().collect()
                        },
                        subarrays,
                        features: 1,
                        items: cumsum,
                    })
                })
                .collect(),
        );
        self.merge_obs(py, &obs[..])
    }

    fn obs_space(&self) -> PyResult<Vec<(String, Vec<String>)>> {
        Ok(self
            .env
            .obs_space
            .entities
            .iter()
            .map(|(k, v)| (k.clone(), v.features.clone()))
            .collect())
    }

    fn action_space(&self) -> PyResult<Vec<(String, Option<Vec<String>>)>> {
        Ok(self
            .env
            .action_space
            .iter()
            .map(|(k, v)| {
                (
                    k.clone(),
                    match v {
                        ActionSpace::Categorical { choices } => Some(choices.clone()),
                        ActionSpace::SelectEntity => None,
                    },
                )
            })
            .collect())
    }

    fn num_envs(&self) -> usize {
        self.env.num_envs
    }
}

impl PyVecEnv {
    fn merge_obs(&self, py: Python, obs: &[Box<Observation>]) -> VecObs {
        let mut raggeds = self
            .env
            .num_feats
            .iter()
            .map(|feats| RaggedBuffer::new(*feats))
            .collect::<Vec<_>>();
        for o in obs {
            let mut start_index = 0;
            for (i, count) in o.features.counts.iter().enumerate() {
                let end_index = start_index + count * raggeds[i].features;
                push(&mut raggeds[i], &o.features.data[start_index..end_index]);
                start_index = end_index;
            }
        }
        let features = self
            .env
            .obs_space
            .entities
            .iter()
            .zip(raggeds)
            .map(|((name, _), ragged)| (name.clone(), RaggedBufferF32(ragged.view())))
            .collect::<Vec<_>>();

        let mut action_masks = FxHashMap::default();
        for (i, (action_name, action_space)) in self.env.action_space.iter().enumerate() {
            let mut ragged_actors = RaggedBuffer::new(1);
            let mut ragged_actees = RaggedBuffer::new(1);
            let mut ragged_mask = RaggedBuffer::new(match action_space {
                ActionSpace::Categorical { choices } => choices.len(),
                ActionSpace::SelectEntity => 0,
            });
            for o in obs {
                match &o.actions[i] {
                    Some(ActionMask::DenseCategorical { actors, mask }) => {
                        push(
                            &mut ragged_actors,
                            &actors.iter().map(|x| *x as i64).collect::<Vec<_>>()[..],
                        );
                        // TODO: could be more efficient, omit mask if None in all obs

                        match mask {
                            Some(mask) => push(
                                &mut ragged_mask,
                                &mask.iter().copied().collect::<Vec<_>>()[..],
                            ),
                            None => {
                                let feats = ragged_mask.features;
                                push(&mut ragged_mask, &vec![true; actors.len() * feats]);
                            }
                        };
                    }
                    Some(ActionMask::SelectEntity { actors, actees }) => {
                        push(
                            &mut ragged_actors,
                            &actors.iter().map(|x| *x as i64).collect::<Vec<_>>()[..],
                        );
                        push(
                            &mut ragged_actees,
                            &actees.iter().map(|x| *x as i64).collect::<Vec<_>>()[..],
                        );
                    }
                    None => match action_space {
                        ActionSpace::Categorical { .. } => {
                            push(&mut ragged_actors, &[]);
                            push(&mut ragged_mask, &[]);
                        }
                        ActionSpace::SelectEntity => {
                            push(&mut ragged_actors, &[]);
                            push(&mut ragged_actees, &[]);
                        }
                    },
                }
            }
            match action_space {
                ActionSpace::Categorical { .. } => {
                    action_masks.insert(
                        action_name.to_string(),
                        (
                            RaggedBufferI64(ragged_actors.view()),
                            None,
                            Some(RaggedBufferBool(ragged_mask.view())),
                        ),
                    );
                }
                ActionSpace::SelectEntity => {
                    action_masks.insert(
                        action_name.to_string(),
                        (
                            RaggedBufferI64(ragged_actors.view()),
                            Some(RaggedBufferI64(ragged_actees.view())),
                            None,
                        ),
                    );
                }
            }
        }

        let reward = obs
            .iter()
            .map(|o| o.reward)
            .collect::<Vec<f32>>()
            .to_pyarray(py)
            .into();
        let done = obs
            .iter()
            .map(|o| o.done)
            .collect::<Vec<bool>>()
            .to_pyarray(py)
            .into();
        let mut metrics = FxHashMap::<_, (usize, f32, f32, f32)>::default();
        for o in obs.iter() {
            for (k, &v) in o.metrics.iter() {
                match metrics.entry(k) {
                    Entry::Occupied(mut e) => {
                        let (count, sum, min, max) = e.get_mut();
                        *count += 1;
                        *sum += v;
                        *min = min.min(v);
                        *max = max.max(v);
                    }
                    Entry::Vacant(e) => {
                        e.insert((1, v, v, v));
                    }
                }
            }
        }

        VecObs {
            features,
            action_masks: self
                .env
                .action_space
                .iter()
                .map(|(name, _)| (name.clone(), action_masks[name].clone()))
                .collect(),
            reward,
            done,
            metrics: metrics.into_iter().map(|(k, m)| (k.clone(), m)).collect(),
        }
    }
}

fn push<T: Copy>(buffer: &mut RaggedBuffer<T>, data: &[T]) {
    buffer
        .subarrays
        .push(buffer.items..(buffer.items + data.len() / buffer.features));
    buffer.data.extend_from_slice(data);
    buffer.items += data.len() / buffer.features;
}
