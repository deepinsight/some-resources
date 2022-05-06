import argparse
import sys
import cv2
import glob
import shutil
import os
import os.path as osp



TMP_DIR = './tmpabc'


def ensure_clear(_dir):
    if osp.exists(_dir):
        shutil.rmtree(_dir)
    os.makedirs(_dir)


def find_valid_dir(input_dir):
    for root, dirs, files in os.walk(input_dir):
        find_first = False
        for _file in files:
            if _file=='001_2.png':
                find_first = True
                break
        if find_first:
            find_all = True
            for i in range(1, 101):
                _file = osp.join(root, "%03d_2.png"%i)
                if not osp.exists(_file):
                    print('image not found:', _file)
                    find_all = False
                    break
                img = cv2.imread(_file)
                if img is None:
                    print('image is none:', _file)
                    find_all = False
                    break
            if not find_all:
                return None
            else:
                return root
    return None


def check_targz(_file):
    assert _file.endswith('.tar.gz')
    result_dir = osp.join(TMP_DIR, 'tarfile')
    ensure_clear(TMP_DIR)
    ensure_clear(result_dir)
    os.system("tar -xzf %s -C %s"%(_file, result_dir))
    result_dir = find_valid_dir(result_dir)
    assert result_dir is not None
    print('result_dir:', result_dir)
    print('success')




if __name__ == "__main__":
    _file = sys.argv[1]
    check_targz(_file)


