# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='delphi',
    version='0.1.0',
    description='Delphi - Integrating Labeling and Learning for Edge-based Visual Data',
    long_description=readme,
    author='Haithem Turki',
    author_email='hturki@cs.cmu.edu',
    url='https://github.com/hturki/delphi',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    entry_points={
        'console_scripts': [
            'learning_module = delphi.learning_module:main',
        ]
    },
)

