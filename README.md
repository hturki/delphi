# Delphi

## Installation steps

```bash
conda env create -n <VIRTUAL_ENV_NAME> -f environment.yml
conda activate <VIRTUAL_ENV_NAME>
python setup.py install
# start the backend
learning_module
```

The virtual env's Python interpreter path is "baked in" in the entry points. So one can also run directly without `conda activate`:
```bash
/path_to_virtual_env/bin/learning_module
```
