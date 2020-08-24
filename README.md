# Delphi

## Installation steps

```bash
conda env create -n <VIRTUAL_ENV_NAME> -f environment.yml
conda activate <VIRTUAL_ENV_NAME>
python setup.py install
# start the backend
learning_module
```

Delphi currently relies on [OpenDiamond](http://diamond.cs.cmu.edu/) to provide examples and static filtering and on [Hyperfind](https://github.com/cmusatyalab/hyperfind) for visualization. The installation steps for these dependencies can be found in their respective repositories.

## Configuration

The server reads from $HOME/.delphi/config.yml at startup by default. The main configuration points are:

- feature_cache: Feature cache configuration (Redis, Filesystem, or Noop)
- root_dir: Data directory where search-relevant data such as training examples are stored
- model_dir: Directory containing pretrained models (normally <repo>/models)
- port: Port the server listens on
