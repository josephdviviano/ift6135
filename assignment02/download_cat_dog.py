#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 15 13:31:59 2018

@author: shawntan
"""

import urllib
import cPickle as pickle
import gzip
import os
import numpy as np
import zipfile
import scipy.ndimage
import glob
import shutil
import warnings
warnings.filterwarnings("ignore")

final_size = 64


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--savedir', type=str, default='datasets',
                        help='directory to save the dataset')
    args = parser.parse_args()

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)
    train_dir = 'train_64x64'
    validation_dir = 'valid_64x64'
    urlpath = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
    filename  = 'train.zip'
    filepath = os.path.join(args.savedir, filename)
    print "Downloading..."
    urllib.urlretrieve(urlpath, filepath)

    print "Extracting file..."
    zip_ref = zipfile.ZipFile(filepath, 'r')
    zip_ref.extractall(args.savedir)
    zip_ref.close()

    train_file_list = glob.glob(os.path.join(
        args.savedir,
        'PetImages',
        '*', '*.jpg'))
    train_proc_data = os.path.join(args.savedir, train_dir)
    valid_proc_data = os.path.join(args.savedir, validation_dir)
    if not os.path.exists(train_proc_data):
        os.makedirs(train_proc_data)
    if not os.path.exists(valid_proc_data):
        os.makedirs(valid_proc_data)


    print "Some files may not open. This is fine."
    for in_pic_path in train_file_list:
        filename = \
            '.'.join(os.path.normpath(in_pic_path).split(os.path.sep)[-2:])
        out_pic_path = os.path.join(train_proc_data, filename)
        try:
            img = scipy.ndimage.imread(in_pic_path)
            side_dim = min(img.shape[0], img.shape[1])
            start_height = (img.shape[0] - side_dim) // 2
            start_width = (img.shape[1] - side_dim) // 2
            img = img[start_height: start_height + side_dim,
                      start_width: start_width + side_dim]

            img = scipy.misc.imresize(
                img,
                size=(final_size, final_size),
                interp='bilinear'
            )

            if len(img.shape) == 3 and img.shape[2] > 3:
                img = img[:, :, 3]

            assert(img.shape[0] == final_size and
                   img.shape[1] == final_size)
            scipy.misc.imsave(out_pic_path, img)
        except IOError:
            print "Could not open", out_pic_path
    for pic_path in glob.glob(os.path.join(args.savedir, train_dir,
                                              '*.1????.jpg')):
        split_path = os.path.normpath(pic_path).split(os.path.sep)
        split_path[-2] = validation_dir
        out_path = os.path.join(*split_path)
        shutil.move(pic_path, out_path)


