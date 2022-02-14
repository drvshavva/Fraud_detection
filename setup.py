import os
import sys
from setuptools import setup, find_packages

if sys.version_info < (3, 7, 7):
    sys.exit('Fraud Detection Python >= 3.7.7')

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))

with open('README.md') as f:
    readme = f.read()

with open("requirements.txt") as f:
    requirements = [i.strip() for i in f.readlines()]

setup(
    name='web_mining_project',
    description='fraud_detection',
    long_description=readme,
    author='havvanur.dervisoglu',
    author_email='drvshavva@gmail.com',
    packages=find_packages(exclude=['*tests*']),
    python_requires='==3.7.7',
    install_requires=requirements,
    entry_points={
        'console_scripts': [
        ]
    },
)