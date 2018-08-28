import os
import pyprind
import sys
import numpy as np

# make directory relative to process working directory
def mkdir_rel(dir):
    dir = os.path.join(os.getcwd(), dir)
    if not os.path.exists(dir):
        try:
            os.mkdir(dir)
        except OSError:
            pass

def pbar(num,time=False):
    return pyprind.ProgBar(num,stream=sys.stdout,bar_char='█',track_time=time)

def param_sched(step,p_max,p_min=0.,n_steps=None,mode='cycle',c_t_ratio=9,exp_base=100,exp_dec_rate=0.99):
    assert mode in ['cycle','exp_dec'], "available 'mode' param values are: 'cycle, 'exp_dec'"
    if mode=='cycle':
        assert n_steps is not None, "n_steps must not be None if mode is 'cycle'"
        cyc_len = np.floor(n_steps * (c_t_ratio / float(c_t_ratio + 1)))//2*2
        stepsize=cyc_len/2
        tail = n_steps - cyc_len
        if cyc_len!=0:
            cycle = np.floor(1 + step / (cyc_len))
            x = np.abs(step / stepsize - 2 * cycle + 1)
        else:
            x=1.0
        p = p_min + (p_max - p_min) * np.max((0.0,1.0 - x))
        if step>=cyc_len:
            x = np.abs((step - float(cyc_len))/ tail)
            new_min=p_min/100.
            p = new_min +(p_min - new_min)*np.max((0.0,1.0-x))
    elif mode=='exp_dec':
        p = p_max*pow(exp_dec_rate,step/exp_base)
    return p