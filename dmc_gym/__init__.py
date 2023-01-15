from gym.envs.registration import register

register(
    id="AcrobotSwingupDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="acrobot", task_name="swingup"),
    max_episode_steps=1_000,
)

register(
    id="AcrobotSwingupSparseDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="acrobot", task_name="swingup_sparse"),
    max_episode_steps=1_000,
)

register(
    id="BallInCupDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="ball_in_cup", task_name="catch"),
    max_episode_steps=1_000,
)

register(
    id="CartpoleDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="cartpole", task_name="balance"),
    max_episode_steps=1_000,
)

register(
    id="CartpoleSparseDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="cartpole", task_name="balance_sparse"),
    max_episode_steps=1_000,
)

register(
    id="CartpoleSwingupDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="cartpole", task_name="swingup"),
    max_episode_steps=1_000,
)

register(
    id="CartpoleSwingupSparseDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="cartpole", task_name="swingup_sparse"),
    max_episode_steps=1_000,
)

register(
    id="CheetahDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="cheetah", task_name="walk"),
    max_episode_steps=1_000,
)

register(
    id="FingerSpinDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="finger", task_name="spin"),
    max_episode_steps=1_000,
)

register(
    id="FingerTurnEasyDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="finger", task_name="turn_easy"),
    max_episode_steps=1_000,
)

register(
    id="FingerTurnHardDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="finger", task_name="turn_hard"),
    max_episode_steps=1_000,
)

register(
    id="FishSwimDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="fish", task_name="swim"),
    max_episode_steps=1_000,
)

register(
    id="FishUprightDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="fish", task_name="upright"),
    max_episode_steps=1_000,
)

register(
    id="HopperDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="hopper", task_name="hop"),
    max_episode_steps=1_000,
)

register(
    id="HopperStandDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="hopper", task_name="stand"),
    max_episode_steps=1_000,
)

register(
    id="HumanoidRunDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="humanoid", task_name="run"),
    max_episode_steps=1_000,
)

register(
    id="HumanoidStandDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="humanoid", task_name="stand"),
    max_episode_steps=1_000,
)

register(
    id="HumanoidWalkDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="humanoid", task_name="walk"),
    max_episode_steps=1_000,
)

register(
    id="ManipulatorDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="manipulator", task_name="bring_ball"),
    max_episode_steps=1_000,
)

register(
    id="PendulumDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="pendulum", task_name="swingup"),
    max_episode_steps=1_000,
)

register(
    id="PointMassDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="point_mass", task_name="easy"),
    max_episode_steps=1_000,
)

register(
    id="ReacherDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="reacher", task_name="easy"),
    max_episode_steps=1_000,
)

register(
    id="ReacherHardDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="reacher", task_name="hard"),
    max_episode_steps=1_000,
)

register(
    id="Swimmer15DMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="swimmer", task_name="swimmer15"),
    max_episode_steps=1_000,
)

register(
    id="Swimmer6DMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="swimmer", task_name="swimmer6"),
    max_episode_steps=1_000,
)

register(
    id="WalkerRunDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="walker", task_name="run"),
    max_episode_steps=1_000,
)

register(
    id="WalkerStandDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="walker", task_name="stand"),
    max_episode_steps=1_000,
)

register(
    id="WalkerDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="walker", task_name="walk"),
    max_episode_steps=1_000,
)


