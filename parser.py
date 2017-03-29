# -*- coding:utf-8 -*-

import argparse

if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataDir", type=str, default="./mnist")
    parser.add_argument("--isTraining", type=int, default=0)
    parser.add_argument("--sampleCount", type=int, default=5)
    flags = parser.parse_args()
    print flags.dataDir
    print flags.isTraining
    print flags.sampleCount
