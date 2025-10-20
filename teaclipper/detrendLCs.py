#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 16:19:08 2025

@author: dario (dario.fritzewski@kuleuven.be)
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev

from .utils import load_lc, save_lc, get_sectors_dates


def all_poly_detrend(TICs, LCdir='./TU/', red_name='', deg=5):
    """
    Apply polynomial detrending to all LCs given in TICs and found in the respective directory.

    Parameters
    ----------
    TICs : array-like or int
        List of TIC IDs to apply detrending to. Single ID is also accepted.
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.
    deg : int, optional
        Polynomial degree of the function used to detrend the LC. The default is 5.

    Returns
    -------
    None.

    """
    if isinstance(TICs, int):
        TICs = [TICs]

    for TIC in TICs:
        try:
            lc = load_lc(TIC, LCdir=LCdir, red_name=red_name)
        except FileNotFoundError:
            print(f'cannot find LC for TIC{TIC}!')
            continue
        dlc = split_and_detrend(lc, deg=deg)
        save_lc(dlc, TIC, LCdir=LCdir, red_name=red_name)


def split_and_detrend(lc, deg=5):
    """
    Get sector border/gaps and apply detrending per piece.
    Parameters
    ----------
    lc : pandas.DataFrame
        Data frame holding the LC.
    deg : int, optional
        Polynomial degree of the function used to detrend the LC. The default is 5.

    Returns
    -------
    dlcdf : pandas.DataFrame
        Data frame holding the detrended LC.

    """

    sector_borders = get_sectors_dates(lc.time.values, split='Contact')

    s_time = np.split(lc.time, sector_borders)
    s_flux = np.split(lc.flux, sector_borders)
    s_fl_err = np.split(lc.flux_err, sector_borders)

    dlc = []
    for i, time in enumerate(s_time):
        # detrending and putting results in data frame
        lcdf = pd.DataFrame(np.array([time, detrend_poly([time, s_flux[i]], deg=deg),
                                      s_fl_err[i]]).T,
                            columns=['time', 'flux', 'flux_err'])
        dlc.append(lcdf)

    dlcdf = pd.concat(dlc)
    return dlcdf


def compareLCs(TIC, LCdir='./TU/', red_name=''):
    """
    Plot a comparison between the raw and detrended LC for interactive comparison

    Parameters
    ----------
    TIC : int
        TIC ID.
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.

    Returns
    -------
    None.

    """
    lc = load_lc(TIC, LCdir=LCdir, red_name=red_name)
    dlc = load_lc(TIC, LCdir=LCdir, red_name=red_name, version='detrended')

    plt.figure()
    plt.plot(lc.time, lc.flux, c='k')
    plt.plot(dlc.time, dlc.flux, c='tab:red')


def detrend_poly(lcpart, deg=5):
    """
    Apply the polynomial detrending.

    Parameters
    ----------
    lcpart : np.array
        2D array of time and flux containing the part of the LC to be detrended.
    deg : int, optional
        Polynomial degree of the function used to detrend the LC. The default is 5.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    cpfit = Chebyshev.fit(lcpart[0], lcpart[1], deg=deg)

    return lcpart[1] - cpfit(lcpart[0])  # dertrendedLC = LC - polynomial
