import os
import pyprind
import sys

# make directory relative to process working directory
def mkdir_rel(dir):
    dir = os.path.join(os.getcwd(), dir)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            pass
def pbar(num):
    return pyprind.ProgBar(num,stream=sys.stdout,bar_char='█')