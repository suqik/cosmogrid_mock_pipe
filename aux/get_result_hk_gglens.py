import numpy as np
import h5py
import sys

# dfile = sys.argv[1]
# rfile = sys.argv[2]
# ofile = sys.argv[3]
# Njk   = int(sys.argv[4])

dfile = "/home/suchen/applications/hk_gglens/result/test_vl_dsigma2.hdf5"
rfile = "/home/suchen/applications/hk_gglens/result/test_vl_dsigma2_rand.hdf5"
ofile = "./aux/results/test_dsigma2_vl.npy"
Njk   = 128

with h5py.File(dfile, "r") as f:
    theta_edge = f['sep_bin'][...] # Mpc/h
    theta_d = 0.5*(theta_edge[1:] + theta_edge[:-1])
    signal_jkf = f['delta_sigma_tan_jkf'][...]
    
with h5py.File(rfile, "r") as f:
    random_jkf = f['delta_sigma_tan_jkf'][...]

JK_prefac = (Njk - 1)/Njk

signal_subtract_jkf = signal_jkf - random_jkf

signal_d = signal_subtract_jkf.mean(axis=0)
signal_err = ((signal_subtract_jkf**2).sum(axis=0) - Njk*signal_d**2)*JK_prefac
signal_err = np.sqrt(signal_err)

np.save(ofile, np.c_[theta_d, signal_d, signal_err])