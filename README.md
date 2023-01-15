# DeepMind Control Suite

## Instalation

```shell
pip install git+https://github.com/qgallouedec/dmc_gym.git
```

## Usage

```python
import gym
import dmc_gym

env = gym.make("CartpoleDMC")

done = False
obs = env.reset()
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
```

## Environments

- `AcrobotSwingupDMC-v0`
- `AcrobotSwingupSparseDMC-v0`
- `BallInCupDMC-v0`
- `CartpoleDMC-v0`
- `CartpoleSparseDMC-v0`
- `CartpoleSwingupDMC-v0`
- `CartpoleSwingupSparseDMC-v0`
- `CartpoleTwoPolesDMC-v0`
- `CartpoleThreePolesDMC-v0`
- `CheetahDMC-v0`
- `FingerSpinDMC-v0`
- `FingerTurnEasyDMC-v0`
- `FingerTurnHardDMC-v0`
- `FishSwimDMC-v0`
- `FishUprightDMC-v0`
- `HopperDMC-v0`
- `HopperStandDMC-v0`
- `HumanoidRunDMC-v0`
- `HumanoidStandDMC-v0`
- `HumanoidWalkDMC-v0`
- `HumanoidRunPureStateDMC-v0`
- `ManipulatorBringBallDMC-v0`
- `ManipulatorBringPegDMC-v0`
- `ManipulatorInsertBallDMC-v0`
- `ManipulatorInsertPegDMC-v0`
- `PendulumDMC-v0`
- `PointMassDMC-v0`
- `PointMassHardDMC-v0`
- `ReacherDMC-v0`
- `ReacherHardDMC-v0`
- `Swimmer15DMC-v0`
- `Swimmer6DMC-v0`
- `WalkerRunDMC-v0`
- `WalkerStandDMC-v0`
- `WalkerDMC-v0`

## Credit

An updated version of https://github.com/denisyarats/dmc2gym