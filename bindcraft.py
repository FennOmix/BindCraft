"""Command-line entry point for running the BindCraft design pipeline."""

import argparse

from functions.generic_utils import perform_input_check
from functions.pipeline import run_bindcraft


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run BindCraft binder design")
    parser.add_argument(
        "--settings",
        "-s",
        type=str,
        required=True,
        help="Path to the basic settings.json file. Required.",
    )
    parser.add_argument(
        "--filters",
        "-f",
        type=str,
        default="./settings_filters/default_filters.json",
        help=(
            "Path to the filters.json file used to filter design. "
            "If not provided, default will be used."
        ),
    )
    parser.add_argument(
        "--advanced",
        "-a",
        type=str,
        default="./settings_advanced/default_4stage_multimer.json",
        help=(
            "Path to the advanced.json file with additional design settings. "
            "If not provided, default will be used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings_path, filters_path, advanced_path = perform_input_check(args)
    run_bindcraft(settings_path, filters_path, advanced_path)


if __name__ == "__main__":
    main()

