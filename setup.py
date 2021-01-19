import sys

if sys.version_info < (3, 6):
    sys.exit('simba requires Python >= 3.6')

from setuptools import setup, find_packages
from pathlib import Path
setup(
    name='simba',
    version='0.1a',
    author='Huidong Chen',
    athor_email='huidong.chen AT mgh DOT harvard DOT edu',
    license='BSD',
    description='SIngle-cell eMBedding Along with features',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='https://github.com/pinellolab/simba',
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        x.strip() for x in
        Path('requirements.txt').read_text('utf-8').splitlines()
    ],
    include_package_data=True,
    package_data={"simba": ["data/*.bed"]}
)
