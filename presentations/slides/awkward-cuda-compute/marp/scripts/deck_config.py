"""Single source of truth for which machine's measured results the deck uses.

The deck's numbers and data-driven figures come from results/<MACHINE>/.
Switch machines with the DECK_MACHINE env var or by editing DEFAULT_MACHINE.

    DECK_MACHINE=B200 python3 scripts/gen_slides.py

The paper references only the B200 results; RTX_A6000 is kept as an archive.
"""
import json
import os

DEFAULT_MACHINE = "RTX_A6000"      # <- change to "B200" once results/B200 is populated
MACHINE = os.environ.get("DECK_MACHINE", DEFAULT_MACHINE)

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "..", "results", MACHINE)


def data(name):
    """Absolute path to a data file inside the active machine's results folder."""
    return os.path.join(RESULTS, name)


def numbers():
    """The scalar measurements (results/<MACHINE>/numbers.json)."""
    path = data("numbers.json")
    if not os.path.exists(path):
        raise SystemExit(
            f"missing {path}\n"
            f"results/{MACHINE}/ is not populated yet. See results/B200/README.md "
            f"to produce it, or set DECK_MACHINE to a populated machine.")
    return json.load(open(path))
