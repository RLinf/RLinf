import init_path
import argparse
import os
import time

import liberopro.liberopro.utils.download_utils as download_utils
from liberopro.liberopro import get_libero_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--download-dir",
        type=str,
        default=get_libero_path("datasets"),
    )
    parser.add_argument(
        "--datasets",
        type=str,
        choices=["all", "libero_goal", "libero_spatial", "libero_object", "libero_100"],
        default="all",
    )
    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use Hugging Face instead of original download links"
    )
    return parser.parse_args()


def main():

    args = parse_args()

    # Ask users to specify the download directory of datasets
    os.makedirs(args.download_dir, exist_ok=True)
    print(f"Datasets downloaded to {args.download_dir}")
    print(f"Downloading {args.datasets} datasets")

    if args.use_huggingface:
        print("Using Hugging Face as the download source")
    else:
        print("Using original download links (note: these may expire soon)")
        input_str = input("Download from original links may lead to failures. Do you want to continue? (y/n): ")
        if input_str.lower() != 'y':
            print("Switching to Hugging Face as the download source...")
            args.use_huggingface = True

    # If not, download
    download_utils.libero_dataset_download(
        download_dir=args.download_dir, 
        datasets=args.datasets,
        use_huggingface=args.use_huggingface
    )


    # wait for 1 second
    time.sleep(1)
    print("\n\n\n")

    # Check if datasets exist first
    download_utils.check_libero_dataset(download_dir=args.download_dir)


if __name__ == "__main__":
    main()
