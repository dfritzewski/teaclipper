#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:59:02 2023

@author: dario (dario.fritzewski@kuleuven.be)
"""
from pathlib import Path
import pandas as pd
import numpy as np
import star_shadow as sts
from . import utils


def prepare_LCs(TICs, LCdir='./TU/', red_name='', version='detrended'):
    """
    Create starshadow compatable light curve files from the Tessutils files.

    Parameters
    ----------
    TICs : list-like
        List-like data structure of TIC IDs.
    LCdir : str, optional
        Root of your Tessutils directory. The default is './TU/'.
    red_name : str, optional
        Additional string attached to the specific reduction of TU.
        The default is ''.
    version : str, optional
        String to add after the default TU file name, i.e. when you did
        additional detrending. The default is 'detrended'.

    Returns
    -------
    Files are saved in a subdirectory of the directory in which LCdir resides.

    """

    if LCdir[-1] != '/':
        LCdir += '/'

    outbase = '/'.join(LCdir.split('/')[:-2]) + '/STS/'
    Path(outbase).mkdir(parents=True, exist_ok=True)

    for TIC in TICs:
        LCdir_l = LCdir + f'stitched{red_name}'
        if version == 'detrended':
            lcname = 'corrected_stitched_detrended'
        else:
            lcname = 'corrected_stitched'
        inputpath = f'{LCdir_l}/lc_tic{TIC}_{lcname}.csv'
        Path(f'{outbase}{red_name}').mkdir(parents=True, exist_ok=True)
        outputpath = f'{outbase}{red_name}/lc_tic{TIC}_corrected_stitched.dat'
        try:
            TU2STS_single(inputpath, outputpath)
        except FileNotFoundError:
            print(inputpath, 'Not found')
            continue


def TU2STS_single(inpath, outpath):
    """
    Reads a TU light curve and saves it as a STS light curve.

    Parameters
    ----------
    inpath : str
        Path of your TU light curve.
    outpath : str
        Path to which the STS light curve should be written.

    Returns
    -------
    None.

    """

    lcdf = pd.read_csv(inpath)

    if np.abs(lcdf.flux.mean()) < 0.1:
        lcdf.flux += 1

    lcdf.dropna(inplace=True)

    lcdf[['time', 'flux', 'flux_err']].to_csv(outpath, sep=' ', header=False,
                                              index=False)


def prewhitening(TIC, indir='./STS/', red_name='', FLOSSY=False):
    """
    Do the pre-whitening with STS and create FLOSSY compatible files.

    Parameters
    ----------
    TIC : int
        TIC ID of star to pre-whiten.
    indir : str, optional
        Directory name where the STS light curves are stored.
        The default is './STS/'.
    red_name : str, optional
        Additional name to distinguish two TU reductions, see prepare_LCs.
        The default is ''.

    Returns
    -------
    Saves all files in their respective directories.

    """

    if indir[-1] != '/':
        indir += '/'

    lcname = f'lc_tic{TIC}_corrected_stitched.dat'
    lcpath = indir + f'{red_name}/' + lcname
    times, signal, _ = np.loadtxt(lcpath, unpack=True)

    # split time into individually handled parts
    border = utils.get_sectors_dates(times, split='Contact')

    i_sectors = []

    # make vector for STS
    for i, t in enumerate(border):
        if i == 0:
            i_sectors.append([0, t])

        if i == len(border) - 1:
            p_i = [t, len(times)]
        else:
            p_i = [t, border[i + 1]]

        i_sectors.append(p_i)

    i_sectors = np.array(i_sectors)

    if len(i_sectors == 0):
        i_sectors = np.array([[0, len(times)]])

    # Do the pre-whitening
    sts.analyse_lc_from_file(lcpath, i_sectors=i_sectors, stage='freq',
                             verbose=False, overwrite=True)

    # plotting
    sts.ut.sequential_plotting(times, signal, i_sectors, lcpath.split('.')[0],
                               indir, save_dir=indir)

    if FLOSSY:
        make_FLOSSY_input(TIC, times, signal, indir=f'{indir}{red_name}/')


def prewhiten_all(TICs, LCdir='./TU/', red_name='', prepare=True, version='detrended',
                  FLOSSY=False):
    """
    Main function to pre-whiten all LC as retrieved from TU.

    Parameters
    ----------
    TICs : list-like
        Holds your TIC IDs.
    LCdir : str, optional
        Root of your Tessutils directory. The default is './TU/'.
    red_name : str, optional
        Additional string attached to the specific reduction of TU.
        The default is ''.
    prepare : bool, optional
        Create STS compatible LCs from TU output. The default is True.
    version : str, optional
        String to add after the default TU file name, i.e. when you did
        additional detrending. The default is 'detrended'.
    FLOSSY : bool
        Create input files for FLOSSY. The default is False.

    Returns
    -------
    All output is saved.

    """
    if prepare:
        prepare_LCs(TICs, LCdir=LCdir, red_name=red_name, version=version)

    indir = '/'.join(LCdir.split('/')[:-2]) + '/STS/'

    for TIC in TICs:
        try:
            prewhitening(TIC, indir=indir, red_name=red_name, FLOSSY=FLOSSY)
        except FileNotFoundError:
            continue


def make_FLOSSY_input(TIC, times, signal,  indir='./STS/'):
    """
    Reads STS output and creates FLOSSY csv files

    Parameters
    ----------
    TIC : int
        TIC ID.
    times : np.array
        Time array of LC.
    signal : np.array
        Flux array of LC.
    indir : TYPE, optional
        Path to the STS extracted signals. The default is './STS/'.

    Returns
    -------
    None.

    """
    if indir[-1] != '/':
        indir += '/'

    lcname = f'lc_tic{TIC}_corrected_stitched'

    filepath = indir + f'{lcname}_analysis/{lcname}_analysis_2.hdf5'

    res = sts.utility.read_parameters_hdf5(filepath)
    len_freq = len(res['sin_mean'][2])

    # make list of pre-whitened frequencies
    flossy_dict = {'frequency': res['sin_mean'][2],
                   'amp': res['sin_mean'][3],
                   'phase': res['sin_mean'][4],
                   'const': np.zeros(len_freq),
                   'period': 1 / res['sin_mean'][2],
                   'e_amp': res['sin_err'][3],
                   'e_frequency': res['sin_err'][2],
                   'e_phase': res['sin_err'][4],
                   'tic': np.ones(len_freq) * TIC,
                   'stopcrit': np.ones(len_freq) * 10}
    df = pd.DataFrame(flossy_dict)

    flossy_name = indir + lcname + '_analysis' + f'/pw_tic{TIC}.csv'

    df[df.frequency > 0.5].to_csv(flossy_name, index=False)

    # make periodogram
    linmodel = sts.timeseries_functions.linear_curve(times, res['sin_mean'][0],
                                                     res['sin_mean'][1],
                                                     res['i_sectors'])
    adj_signal = signal - linmodel

    f, a = sts.timeseries_functions.astropy_scargle(times, adj_signal)

    fsmall = np.where(f > 0.5)

    pg_name = indir + lcname + '_analysis' + f'/pg_tic{TIC}.csv'
    np.savetxt(pg_name, np.array([f[fsmall], a[fsmall]]).T, delimiter=',',
               header='freq,amp', comments='')
