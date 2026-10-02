"""DEPRECATED — NOT FOR FORMAL LAGPER EXPERIMENTS.

The former script randomly split AG-ReID identities and mislabeled the result
as LAGPeR. That behavior has been removed. A correct converter requires the
real LAGPeR raw seven-scene tree and its official metadata, which are not
present in this repository.
"""

import argparse


MESSAGE = (
    "BLOCKED: NEED REAL LAGPER DATA TREE. prepare_lag.py is deprecated and "
    "cannot create the official LAGPeR scene split. Supply a prepared tree "
    "matching docs/rahp-cesa-technical-design.md or provide the real raw "
    "seven-scene layout and official metadata."
)


def main():
    parser = argparse.ArgumentParser(
        description='Deprecated non-official LAGPeR converter')
    parser.parse_args()
    raise SystemExit(MESSAGE)


if __name__ == '__main__':
    main()
