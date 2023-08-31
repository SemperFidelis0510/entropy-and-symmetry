from source.main import *
from datetime import datetime


def run_tests():  # To be finished
    dst_folder = f'../entropy_results/testing={datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    src_folder = "../datasets/noising"
    main(dst_folder=dst_folder, src_folder=src_folder)


if __name__ == "__main__":
    run_tests()
