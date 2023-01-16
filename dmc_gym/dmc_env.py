from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from dm_control import suite
from dm_env import TimeStep, specs
from gym import Env, spaces


def extract_min_max(spec: Union[specs.Array, specs.BoundedArray]) -> Tuple[float, float]:
    """
    Extract minimum and maximum values from a given specification.

    Parameters:
        spec (specification object): A specification object

    Returns:
        Tuple: A tuple of minimum and maximum values
    """
    assert spec.dtype == np.float64 or spec.dtype == np.float32
    dim = int(np.prod(spec.shape))
    if isinstance(spec, specs.BoundedArray):
        low, high = spec.minimum, spec.maximum
    else:
        assert isinstance(spec, specs.Array)
        low, high = -np.inf, np.inf

    return (low * np.ones(dim)).astype(np.float32), (high * np.ones(dim)).astype(np.float32)


def spec_to_box(specs: List[Union[specs.Array, specs.BoundedArray]]) -> spaces.Box:
    """
    Convert a given specification to a Box Space.

    Parameters:
        specs (List): A list of specifications.
        dtype (Type): The desired data type for the returned Box Space.

    Returns:
        Box: A Box Space object with the specified data type.

    """
    lows, highs = [], []
    for spec in specs:
        low, high = extract_min_max(spec)
        lows.append(low)
        highs.append(high)
    lows, highs = np.concatenate(lows), np.concatenate(highs)
    return spaces.Box(lows, highs, dtype=np.float32)


def flatten_dict_observation(obs: Dict[str, Union[float, np.ndarray]]) -> np.ndarray:
    """
    Flatten the observation dictionary and return a 1-D numpy array.

    Parameters:
        obs (dict): A dictionary containing the observations.

    Returns:
        np.ndarray: A 1-D numpy array containing the flattened observations.
    """
    flattened_obs = [np.array([val]) if np.isscalar(val) else np.ravel(val) for val in obs.values()]
    return np.concatenate(flattened_obs)


class DMCEnv(Env):
    """
    Deepmind Control Suite Envrionment.

    This class is a wrapper for integrating Deepmind Control Suite environment with gym.
    See https://github.com/deepmind/dm_control.

    Args:
        domain_name (str): Domain name
        task_name (str): Task name
        from_pixels (bool, optional): If True, the observation is the
            image from the scene. If False, it's the state as provided
            by dm_control. Defaults to False.
    """

    metadata = {"render.modes": ["rgb_array"]}

    def __init__(self, domain_name: str, task_name: str, from_pixels: bool = False) -> None:
        self._from_pixels = from_pixels

        # Create task
        self._env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs={"random": [42]})

        # Create action space
        self.action_space = spec_to_box([self._env.action_spec()])

        # Create observation space
        if from_pixels:
            self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 3), dtype=np.uint8)
        else:
            self.observation_space = spec_to_box(self._env.observation_spec().values())

        self.reward_range = (0, 1)
        self.metadata["video.frames_per_second"] = int(1 / self._env.physics.model.opt.timestep)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._env, name)

    def seed(self, seed: Optional[int]) -> None:
        self.action_space.seed(seed)
        self.observation_space.seed(seed)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        time_step = self._env.step(action)  # type: TimeStep
        if self._from_pixels:
            obs = self._env.physics.render(height=84, width=84, camera_id=0)
        else:
            obs = flatten_dict_observation(time_step.observation).astype(np.float32)
        reward = time_step.reward
        done = time_step.last()
        info = {"internal_state": self._env.physics.get_state(), "discount": time_step.discount}
        return obs, reward, done, info

    def reset(self) -> np.ndarray:
        time_step = self._env.reset()
        if self._from_pixels:
            obs = self._env.physics.render(height=84, width=84, camera_id=0)
        else:
            obs = flatten_dict_observation(time_step.observation).astype(np.float32)
        return obs

    def render(self, mode: int = "rgb_array") -> np.ndarray:
        if mode == "rgb_array":
            return self._env.physics.render(height=480, width=480, camera_id=0)
        elif mode == "human":
            raise NotImplementedError("Help welcomed.")
