"""Usage: python -m python.train.single_frame --data-dir ... --output-dir ..."""

from .engine import build_parser, run


if __name__ == "__main__":
    run(build_parser(sequence=False).parse_args())
