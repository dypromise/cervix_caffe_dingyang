import sys
import lmdb
import numpy as np
from argparse import ArgumentParser
python_dir='/mnt/lustre/dingyang/cervix_caffe_dingyang/sensenet-release/core/python'
if python_dir not in sys.path:
    sys.path.insert(0, python_dir)
from caffe.proto.caffe_pb2 import Datum


def main(args):
    datum = Datum()
    data = []
    env = lmdb.open(args.input_lmdb)
    with env.begin() as txn:
        cursor = txn.cursor()
        for i, (key, value) in enumerate(cursor):
            if i >= args.truncate: break
            datum.ParseFromString(value)
            data.append(datum.float_data)
    data = np.squeeze(np.asarray(data))
    np.save(args.output_npy, data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_lmdb')
    parser.add_argument('output_npy')
    parser.add_argument('--truncate', type=int, default=np.inf,
            help="Stop converting the items from the database after this. "
                 "All the items will be converted if not specified.")
    args = parser.parse_args()
    main(args)
