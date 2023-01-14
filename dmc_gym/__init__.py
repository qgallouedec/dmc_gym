from gym.envs.registration import register

register(
    id="PointMassDMC-v0",
    entry_point="dmc_gym.wrappers:DMCWrapper",
    kwargs=dict(domain_name="point_mass", task_name="easy"),
    max_episode_steps=1_000,
)
