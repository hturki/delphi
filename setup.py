# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_call

with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()


class PreInstallCommand(install):
    """Pre-installation for installation mode."""

    def run(self):
        proto_root = Path.cwd() / 'protos'
        proto_dir = proto_root / 'delphi' / 'proto'
        for proto_file in proto_dir.iterdir():
            check_call(
                'python -m grpc_tools.protoc -I{} {} --python_out=. --grpc_python_out=. --mypy_out=.'
                .format(proto_root, proto_file)
                .split())

        install.run(self)


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
    cmdclass={
        'install': PreInstallCommand,
    },
)
