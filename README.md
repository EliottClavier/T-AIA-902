# T-AIA-902-NAN_1

## Install requirements

- [Python 3.12](https://www.python.org/downloads/)

Run the installer

- [pipx](https://pipx.pypa.io/)

```console
$ py -m pip install --user pipx
$ py -m pipx ensurepath
```

- [poetry](https://python-poetry.org/)

```console
$ py -m pipx install poetry
```

## Run project

In [boostrap](boostrap) or [taxi_driver](taxi_driver) directory, run:

```console
$ poetry install
```

To start the program:

```console
$ poetry run python <file>.py
# or activate the shell
$ poetry shell
$ py <file>.py
```
