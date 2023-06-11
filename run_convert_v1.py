import argparse
from convert.v1 import convert
# memanggil convert all v1
# memanggil convert benign v1
# memanggil convert mirai v1

if __name__ == '__main__':
    desc = 'Convert botnet dataset into image'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-f', '--filter_column', default=None, help='filter query for dataset columns')
    parser.add_argument('-d', '--dataset_filename', required=True, help='dataset filename include extension')
    args = parser.parse_args()

    convert(dataset_filename=args.dataset_filename, filter_column=args.filter_column)