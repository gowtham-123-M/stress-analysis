"""
	Date : 11/10/2022

	AboutFile:
		> Features : Time-domain features , Frequency-domain features
		> Here we will build the features for the dhyana inference.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.integrate import trapz
from scipy import signal


def get_time_domain_features(rr_intervals, pnni_as_percent = True):

	"""
		Features :
			> MEAN_RR
			> SDRR
			> RMSSD
			> SDSD
			> pNN25
			> pNN50
	"""

	diff_rri = np.diff(rr_intervals)
	length_int = len(rr_intervals)-1 if pnni_as_percent else len(rr_intervals)

	## Basic statistics
	MEAN_RR = np.mean(rr_intervals)
	median_rri = np.median(rr_intervals)
	range_rri = max(rr_intervals)- min(rr_intervals)

	SDSD = np.std(diff_rri) ## standard deviation of difference of rr intervals
	RMSSD = np.sqrt(np.mean(diff_rri**2)) ## Root Mean square of difference.

	## pNN25, pNN50
	nni_50 = sum(np.abs(diff_rri) > 50)
	pNN50 = 100*nni_50/length_int

	nni_25 = sum(np.abs(diff_rri) > 25)
	pNN25 = 100*nni_25/length_int


	## SDRR
	SDRR = np.std(rr_intervals, ddof=1)

	'''
		Heart Rate Equivalent Features
		This space is for the heart rate features 
		heart_rate, mean_hr, min_hr, max_hr, std_hr 
	'''
	td_features = {
		"MEAN_RR" : MEAN_RR,
		"SDRR" : SDRR,
		"RMSSD" : RMSSD,
		"SDSD" : SDSD,
		"pNN25" : pNN25,
		"pNN50" : pNN50
	}
	return td_features



def get_frequency_domian_features(rr_intervals, fs = 4):

	"""
		Frequency Domain Features
			> LF
			> HF
			> LF_HF

		Frequencies in the bands
			> VLF : 0-0.04Hz
			> LF : 0.04 - 0.15Hz
			> HF : 0.15 - 0.4Hz
	"""

	## getting the spectral density using Welch's method
	fxx, pxx = signal.welch(rr_intervals, fs=fs)
	cond_vlf = (fxx >= 0) & (fxx < 0.04)
	cond_lf = (fxx >= 0.04) & (fxx < 0.15)
	cond_hf = (fxx >= 0.15) & (fxx < 0.4)

	## calculating the power band by a Area Under the Curve (AUC) we can get by the Integration.
	vlf = trapz(pxx[cond_vlf], fxx[cond_vlf])
	LF = trapz(pxx[cond_lf], fxx[cond_lf])
	HF = trapz(pxx[cond_hf], fxx[cond_hf])

	fd_features = {
		"LF" : LF,
		"HF" : HF,
		"LF_HF" : LF/HF
	}
	return fd_features

