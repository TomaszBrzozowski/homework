import os

# make directory relative to process working directory
def mkdir_rel(dir):
    dir = os.path.join(os.getcwd(), dir)
    try:
        os.mkdir(dir)
    except OSError:
        pass