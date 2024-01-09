from setuptools import setup, find_packages
import glob

scripts = glob.glob('bin/*')
scripts = [s for s in scripts if '~' not in s]

setup(
    name="atm_psf",
    description="tools for simulating atmospheric psfs",
    version="0.1.0",
    packages=find_packages(),
    scripts=scripts,
    license="GPL",
    author="Erin Sheldon",
    url="https://github.com/esheldon/atm_psf",
)
