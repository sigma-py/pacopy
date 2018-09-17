# pacopy

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/pacopy/master.svg)](https://circleci.com/gh/nschloe/pacopy/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pacopy.svg)](https://codecov.io/gh/nschloe/pacopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/pacopy.svg)](https://pypi.org/project/pacopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pacopy.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/pacopy)

[Numerical continuation](https://en.wikipedia.org/wiki/Numerical_continuation) in Python.

### Examples

#### Ginzburg–Landau

![ginzburg-landau](https://nschloe.github.io/pacopy/ginzburg-landau.gif)

The [Ginzburg-Landau
equations](https://en.wikipedia.org/wiki/Ginzburg%E2%80%93Landau_theory) model the
behavior of extreme type-II superconductors under a magnetic field. The above example
(to be found in full detail
[here](https://github.com/nschloe/pacopy/blob/master/test/test_ginzburg_landau.py))
shows parameter continuation in the strength of the magnetic field. The plot on the
right-hand side shows the absolute value of the complex-valued solution.


### Installation

pacopy is [available from the Python Package
Index](https://pypi.org/project/pacopy/), so simply type
```
pip install -U pacopy
```
to install or upgrade.

### Testing

To run the pacopy unit tests, check out this repository and type
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

pacopy is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
