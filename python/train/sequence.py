"""Usage: python -m python.train.sequence --checkpoint ... --data-dir ... --output-dir ..."""

from .engine import build_parser, run


if __name__ == "__main__":
    run(build_parser(sequence=True).parse_args())
