import argparse
from config import BENIGN_FILENAME, MIRAI_FILENAME
from convert.v2 import convert


if __name__ == '__main__':
    desc = 'Convert botnet dataset into image using feature selection'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--filter_column', default=None, help='filter query for dataset columns')
    parser.add_argument('-t', '--threshold', required=True, type=float, help='threshold value for correlation')
    args = parser.parse_args()

    convert(
        benign_filename=BENIGN_FILENAME,
        mirai_filename=MIRAI_FILENAME,
        filter_column=args.filter_column,
        threshold=args.threshold
    )