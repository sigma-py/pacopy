# pacopy

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/pacopy/master.svg)](https://circleci.com/gh/nschloe/pacopy/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/pacopy.svg)](https://codecov.io/gh/nschloe/pacopy)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![Documentation Status](https://readthedocs.org/projects/pacopy/badge/?version=latest)](https://pacopy.readthedocs.org/en/latest/?badge=latest)
[![PyPi Version](https://img.shields.io/pypi/v/pacopy.svg)](https://pypi.org/project/pacopy)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/pacopy.svg?logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/pacopy)

pacopy provides various algorithms of [numerical parameter
continuation](https://en.wikipedia.org/wiki/Numerical_continuation) for PDEs in Python.

pacopy is backend-agnostic, so it doesn't matter if your problem is formulated with
[SciPy](https://www.scipy.org/), [FEniCS](https://fenicsproject.org/),
[pyfvm](https://github.com/nschloe/pyfvm), or any other Python package. The only thing
the user must provide is a class with some simple methods, e.g., a function evaluation
`f(u, lmbda)`, a Jacobian a solver `jac_solver(u, lmbda, rhs)` etc.

Some pacopy documentation is available [here](https://pacopy.readthedocs.org/en/latest/?badge=latest).


### Examples

#### Bratu

<img src="https://nschloe.github.io/pacopy/bratu1d.png" width="30%">

The classical [Bratu
problem](https://en.wikipedia.org/wiki/Liouville%E2%80%93Bratu%E2%80%93Gelfand_equation)
in 1D with Dirichlet boundary conditions. To reproduce the plot, you first have to
specify the problem; this is the classical finite-difference approximation:
```python
class Bratu1d(object):
    def __init__(self):
        self.n = 51
        h = 1.0 / (self.n - 1)

        self.H = numpy.full(self.n, h)
        self.H[0] = h / 2
        self.H[-1] = h / 2

        self.A = (
            scipy.sparse.diags([-1.0, 2.0, -1.0], [-1, 0, 1], shape=(self.n, self.n))
            / h ** 2
        )
        return

    def inner(self, a, b):
        """The inner product of the problem. Can be numpy.dot(a, b), but factoring in
        the mesh width stays true to the PDE.
        """
        return numpy.dot(a, self.H * b)

    def norm2_r(self, a):
        """The norm in the range space; used to determine if a solution has been found.
        """
        return numpy.dot(a, a)

    def f(self, u, lmbda):
        """The evaluation of the function to be solved
        """
        out = self.A.dot(u) - lmbda * numpy.exp(u)
        out[0] = u[0]
        out[-1] = u[-1]
        return out

    def df_dlmbda(self, u, lmbda):
        """The function's derivative with respect to the parameter. Used in Euler-Newton
        continuation.
        """
        out = -numpy.exp(u)
        out[0] = 0.0
        out[-1] = 0.0
        return out

    def jacobian_solver(self, u, lmbda, rhs):
        """A solver for the Jacobian problem.
        """
        M = self.A.copy()
        d = M.diagonal().copy()
        d -= lmbda * numpy.exp(u)
        M.setdiag(d)
        # Dirichlet conditions
        M.data[0][self.n - 2] = 0.0
        M.data[1][0] = 1.0
        M.data[1][self.n - 1] = 1.0
        M.data[2][1] = 0.0
        return scipy.sparse.linalg.spsolve(M.tocsr(), rhs)
```
Then pass the object to any of pacopy's methods, e.g., the Euler-Newton (arclength)
continuation:
```python
problem = Bratu1d()
# Initial guess
u0 = numpy.zeros(problem.n)
# Initial parameter value
lmbda0 = 0.0

lmbda_list = []
values_list = []
def callback(k, lmbda, sol):
    # Use the callback for plotting, writing data to files etc.
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel("$\\lambda$")
    ax1.set_ylabel("$||u||_2$")
    ax1.grid()

    lmbda_list.append(lmbda)
    values_list.append(numpy.sqrt(problem.inner(sol, sol)))

    ax1.plot(lmbda_list, values_list, "-x", color="#1f77f4")
    ax1.set_xlim(0.0, 4.0)
    ax1.set_ylim(0.0, 6.0)
    return

# Natural parameter continuation
# pacopy.natural(problem, u0, lmbda0, callback, max_steps=100)

pacopy.euler_newton(
    problem, u0, lmbda0, callback, max_steps=500, newton_tol=1.0e-10
)
```


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
