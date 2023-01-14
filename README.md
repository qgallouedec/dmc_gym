# DeepMind Control Suite

## Instalation

```shell
pip install git+https://github.com/qgallouedec/dmc_gym.git
```

## Usage

```python
import gym
import dmc_gym

env = gym.make("PointMass-v0")

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

## Credit

An updated version of https://github.com/denisyarats/dmc2gym