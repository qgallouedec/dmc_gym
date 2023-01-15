from gym.envs.registration import register

register(
    id="AcrobotSwingupDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="acrobot", task_name="swingup"),
    max_episode_steps=1_000,
)


register(
    id="PointMassDMC-v0",
    entry_point="dmc_gym.dmc_env:DMCEnv",
    kwargs=dict(domain_name="point_mass", task_name="easy"),
    max_episode_steps=1_000,
)
