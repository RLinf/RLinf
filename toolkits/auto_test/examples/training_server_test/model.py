"""Idle resources training service example."""

import argparse


def train():
    """Training entry placeholder."""
    print("training_server_test")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="training_server_test")
    parser.add_argument("command", nargs="?", default="train")
    args = parser.parse_args()

    if args.command == "train":
        train()
