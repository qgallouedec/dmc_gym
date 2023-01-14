from setuptools import find_packages, setup

setup(
    name="dmc_gym",
    version="0.0.0",
    author="Quentin Gallouédec",
    description=("Deepmind Control Suite"),
    license="MIT",
    packages=find_packages(),
    install_requires=["gym==0.21", "dm_control"],
)
