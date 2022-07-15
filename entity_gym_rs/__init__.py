from typing import Any, Dict, Mapping
import numpy.typing as npt
import numpy as np
from ragged_buffer import RaggedBufferI64, RaggedBufferF32, RaggedBufferBool
from entity_gym.env import (
    VecEnv,
    Entity,
    ActionSpace,
    CategoricalActionSpace,
    SelectEntityActionSpace,
    ObsSpace,
    VecObs,
    ActionName,
    VecCategoricalActionMask,
    VecSelectEntityActionMask,
)


def clean_ragged_i64(x: RaggedBufferI64) -> RaggedBufferI64:
    return RaggedBufferI64.from_flattened(
        flattened=x.as_array(),
        lengths=x.size1(),
    )


def clean_ragged_f32(x: RaggedBufferI64) -> RaggedBufferI64:
    return RaggedBufferF32.from_flattened(
        flattened=x.as_array(),
        lengths=x.size1(),
    )


def clean_ragged_bool(x: RaggedBufferBool) -> RaggedBufferBool:
    return RaggedBufferBool.from_flattened(
        flattened=x.as_array(),
        lengths=x.size1(),
    )


def to_vec_obs(x) -> VecObs:
    action_masks = {}
    for action_name, (actors, actees, mask) in x.action_masks:
        if actees is None:
            action_masks[action_name] = VecCategoricalActionMask(
                actors=clean_ragged_i64(actors), mask=clean_ragged_bool(mask)
            )
        else:
            action_masks[action_name] = VecSelectEntityActionMask(
                actors=clean_ragged_i64(actors),
                actees=clean_ragged_i64(actees),
            )

    return VecObs(
        features={k: clean_ragged_f32(v) for k, v in x.features.items()},
        visible={},
        action_masks=action_masks,
        reward=x.reward,
        done=x.done,
        metrics={},
    )


class RustVecEnv(VecEnv):
    def __init__(self, env: Any) -> None:
        self._env = env

    def obs_space(self) -> ObsSpace:
        return ObsSpace(
            entities={
                name: Entity(feats) for name, feats in self._env.obs_space().items()
            },
        )

    def action_space(self) -> Dict[ActionName, ActionSpace]:
        return {
            name: CategoricalActionSpace(labels)
            if labels is not None
            else SelectEntityActionSpace()
            for name, labels in self._env.action_space()
        }

    def reset(self, obs_config: ObsSpace) -> VecObs:
        return to_vec_obs(self._env.reset())

    def act(
        self, actions: Mapping[ActionName, RaggedBufferI64], obs_filter: ObsSpace
    ) -> VecObs:
        return to_vec_obs(
            self._env.act([(a.as_array(), a.size1()) for _, a in actions.items()])
        )

    def render(self, **kwargs: Any) -> npt.NDArray[np.uint8]:
        raise NotImplementedError

    def __len__(self) -> int:
        return self._env.num_envs()
