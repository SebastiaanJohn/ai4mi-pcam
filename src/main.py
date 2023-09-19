"""Main module for the project."""

from argparse import ArgumentParser


def main(args) -> None:
    """Main method for the project."""



if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the directory containing the data.",
    )

    args = parser.parse_args()

    main(args)
