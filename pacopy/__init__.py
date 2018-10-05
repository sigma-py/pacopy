# -*- coding: utf-8 -*-
#
from __future__ import print_function

from .__about__ import __author__, __email__, __license__, __version__, __status__

from .natural import natural
from .euler_newton import euler_newton
from .branch_switching import branch_switching

__all__ = [
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "__status__",
    "natural",
    "euler_newton",
    "branch_switching",
]

# try:
#     import pipdate
# except ImportError:
#     pass
# else:
#     if pipdate.needs_checking(__name__):
#         print(pipdate.check(__name__, __version__), end="")
