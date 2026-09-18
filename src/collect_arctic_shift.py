"""Compatibility entry point for the Reddit Arctic Shift collector."""

try:
    from src.platforms.reddit.cli import main
except ModuleNotFoundError:
    from platforms.reddit.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
