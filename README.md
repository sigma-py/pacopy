# pycont

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/pycont/master.svg)](https://circleci.com/gh/nschloe/pycont/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pycont.svg)](https://codecov.io/gh/nschloe/pycont)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/pycont.svg)](https://pypi.org/project/pycont)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pycont.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/pycont)

[Numerical continuation](https://en.wikipedia.org/wiki/Numerical_continuation) in Python.

### Installation

pycont is [available from the Python Package
Index](https://pypi.org/project/pycont/), so simply type
```
pip install -U pycont
```
to install or upgrade.

### Testing

To run the pycont unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

pycont is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
