"""``python -m pytrosna`` runs the command-line tool."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
