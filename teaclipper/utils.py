#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 30 17:13:01 2025

@author: dario (dario.fritzewski@kuleuven.be)

Holding all kinds of utility functions that are required for several steps.

"""
from pathlib import Path
import requests
import pandas as pd
import numpy as np
from astropy.time import Time
try:
    import lightkurve as lk
    lk_check = True
except ModuleNotFoundError:
    print('Lightkurve not installed. Can read light curves only into np.arrays')
    lk_check = False


def load_lc(TIC, LCdir='./TU/', version='TU', red_name='', return_lk=False):
    """
    Load a TESSutils light curve either raw or detrended.

    Parameters
    ----------
    TIC : int
        TIC IDs of your target.
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    version : str, optional
        Which version of the light curve to read. Either 'TU' meaning the raw LC as produced
        by TESSutils, or 'detrended' meaining the LC after polynomial detrending. The default
        is 'TU'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.
    return_lk : bool, optional
        Return lightkurve.LightCurve. The default is False.

    Returns
    -------
    lcdf : pandas.DataFrame
        Your light curve as a pandas data frame.
    lc : lightkurve.LightCurve
        If return_lk the return is a lightkurve.LightCurve object.

    """

    if version == 'TU':
        lcname = '_corrected_stitched'
    elif version == 'detrended':
        lcname = '_corrected_stitched_detrended'
    else:
        lcname = '_corrected_stitched'

    LC_dir = make_dir_path(LCdir, red_name)
    lcpath = f'{LC_dir}lc_tic{TIC}{lcname}.csv'

    lcdf = pd.read_csv(lcpath)
    lcdf.dropna(inplace=True)  # remove NaNs

    if return_lk & lk_check:
        lc = lk.LightCurve(time=lcdf.time, flux=lcdf.flux, flux_err=lcdf.flux_err)
        return lc
    return lcdf


def save_lc(lc, TIC, LCdir='./TU/', red_name=''):
    """
    Save the detrended light curve to the same directory as your original LC.

    Parameters
    ----------
    lc : pandas.DataFrame
        Data frame holding the detrended LC.
    TIC : int
        TIC ID of the LC.
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.

    Returns
    -------
    None.

    """

    LC_dir = make_dir_path(LCdir, red_name)

    lcname = '_corrected_stitched_detrended'
    lcpath = f'{LC_dir}lc_tic{TIC}{lcname}.csv'

    lc.to_csv(lcpath, index=False)


def get_sectors(lc, limit=0.5):
    """
    Get indices of gaps/sector borders based on a simple gap search. To split at the
    actual Sector/Orbit/Contact points use get_sectors_dates

    Parameters
    ----------
    lc : pandas.DataFrame
        Data frame holding the light curve.
    limit : float, optional
        Width of gap required to be considered a gap/sector border in days. The default is 0.5.

    Returns
    -------
    np.array
        List of indices at which gaps appear.

    """

    timediffs = np.diff(lc.time)  # get time differences between consecutive data points
    jumps = np.where(timediffs > limit)[0]  # find indices of larger gaps

    if len(jumps) == 0:
        return 1

    return jumps + 1  # adjust for diff shift of 1


def make_dir_path(LCdir='./TU/', red_name=''):
    """
    Create the path name from the given information.

    Parameters
    ----------
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.

    Returns
    -------
    LCdir : str
        The complete path name to the LC directory.

    """
    if LCdir[-1] != '/':
        LCdir += '/'

    LCdir += f'stitched{red_name}/'

    return LCdir


def get_sectors_dates(time, Sectors='TESS_orbit_times.csv', split='Sector', return_numbers=False):
    """
    Get indices of Sector borders to split light curve for detrending or plotting.

    Parameters
    ----------
    time : np.array-like
        Vector of all time stamps of the light curve.
    Sectors : str, optional
        Path to the TESS_orbit_times.csv overview file.

        If your LC has times later than the latest datum in the file it will be
        downloaded. At the time of writing this file is updated regularly here:
        https://tess.mit.edu/public/files/TESS_orbit_times.csv.

        The default is 'TESS_orbit_times.csv'.
    split : str, optional
        Indicates the position of the splitting. Can be from ['Sector', 'Orbit', 'Contact'].
        'Sector': ~27 d-long stretches, unless you are dealing with some of the longer TESS sectors.
        'Orbit': One TESS sector is typically made up of two orbits.
        'Contact': Splits the LC at every scheduled down-link gap. Same as Orbit for Sectors < 56.
        The default is 'Sector'.
    return_number : bool, optional
        Return the sector numbers for all segments. The default is False.

    Returns
    -------
    border : np.array
        Indices of time that conincide to the borders between the Sectors/orbits.
    sector : np.array, optional
        Sector numbers corresponding to the observed times.

    """

    if split not in ['Sector', 'Orbit', 'Contact']:
        raise ValueError("split must be in  ['Sector', 'Orbit', 'Contact']")

    TESSzero = 2457000.0  # zero point of TESS time stamps
    T0 = Time(str(TESSzero), format='jd', scale='utc')

    # make sure the path always points to the file within the package
    p = Path(__file__).with_name(Sectors)
    fname = p.absolute()

    orbitt = pd.read_csv(fname).dropna()

    endofSectors = Time(orbitt['End of Orbit'].iloc[-1], format='iso', scale='utc') - T0

    # check if Sectors file is up-to-date
    if check_sector_file(time[-1], endofSectors.jd):

        orbitt = pd.read_csv(fname).dropna()

    # transform to JD
    orbitt['startJD'] = orbitt['Start of Orbit'].apply(Time, format='iso', scale='utc')
    orbitt['endJD'] = orbitt['End of Orbit'].apply(Time, format='iso', scale='utc')

    rows = []

    # make a df of the requested times
    if split == 'Contact':
        timedf = orbitt[['startJD', 'endJD']].copy()
        for col in timedf.columns:
            timedf[col] = timedf[col].apply(lambda x: (x - T0).jd)
        timedf['Sector'] = orbitt['Sector']
    else:  # Sector or Contact
        for _, group in orbitt.groupby(split):
            tmin = (group.startJD.min() - T0).jd
            tmax = (group.endJD.max() - T0).jd
            segment_num = group['Sector'].iloc[0]
            rows.append([tmin, tmax, segment_num])

        timedf = pd.DataFrame(rows, columns=['startJD', 'endJD', 'Sector'])

    # match times to LC
    # 1. find range of sectors
    tmin = np.nanmin(time)
    tmax = np.nanmax(time)

    borders = []
    sector_numbers = []  # sector *ending* at this border

    # 2. find for each Sector the minimum distant data point in time (its index)
    for _, row in timedf[timedf.endJD.between(tmin, tmax)].iterrows():
        b = np.searchsorted(time, row.endJD+0.1)  # add 0.1 to avoid spill over into next sector
        if b in borders:  # avoid repeated value due to missing Sectors
            continue
        if b - 1 in borders:  # avoid the same for the final missing sector
            continue
        borders.append(b)
        sector_numbers.append(int(row['Sector']))

    if len(borders) == 0:
        no_border = np.array([], dtype=int)
        if return_numbers:
            covering = timedf[(timedf.startJD <= tmin) & (timedf.endJD >= tmax)]
            nums = [int(covering['Sector'].iloc[0])] if not covering.empty else []
            return no_border, nums
        return no_border

    # add the final sector number safely
    idx_check = min(borders[-1] + 1, len(time) - 1)
    final_rows = timedf[(timedf.startJD < time[idx_check]) & (timedf.endJD >= tmax)]

    if not final_rows.empty:
        final_sector = int(final_rows['Sector'].iloc[0])
        sector_numbers.append(final_sector)
    elif len(sector_numbers) > 0:
        sector_numbers.append(sector_numbers[-1])
    else:
        sector_numbers.append(1)  # Fallback sector number if nothing matches

    # sort independently is possible as both follw the same time line
    borders = np.sort(np.array(borders))
    sector_numbers = np.sort(np.array(sector_numbers))

    if return_numbers:
        return borders, sector_numbers
    return borders


def check_sector_file(tmaxlc, tmaxTESS):
    '''
    Check if your TESS_orbit_times.csv is up-to-date.

    Parameters
    ----------
    tmaxlc : float
        MJD of the last datum in the light curve.
    tmaxTESS : float
        MJD of the entry in TESS_orbit_times.csv.

    Returns
    -------
    bool
        Indicates whether the file has been updated.

    '''
    if tmaxlc > tmaxTESS:
        # download the latest version
        url = 'https://tess.mit.edu/public/files/TESS_orbit_times.csv'
        p = Path(__file__).with_name('TESS_orbit_times.csv')
        fname = p.absolute()

        with open(fname, 'wb') as out_file:
            content = requests.get(url, stream=True, timeout=10).content
            out_file.write(content)
        return True

    return False


def make_ticks(ax):
    '''
    Turn on the minor and right/top ticks

    Parameters
    ----------
    ax : matplotlib.axis
        The axis object that should be modified.
    '''
    ax.tick_params(top=True, right=True, which='both')
    ax.minorticks_on()


def FLOSSY2amigo(TIC, directory, addpart='', FLOSSYsubdir='pleiades'):
    '''
    Create AMiGO compatible files with patterns found through FLOSSY

    Parameters
    ----------
    TIC : int
        TIC ID.
    directory : str
        Path to FLOSSY dir.
    addpart : str, optional
        Additional part in file at end to distinguish two patterns. The default is ''.
    FLOSSYsubdir : str, optional
        Name of subdirectory within FLOSSY where the current data is stored.
        The default is 'pleiades'.

    Returns
    -------
    None.

    '''

    lcname = f'lc_tic{TIC}_corrected_stitched'
    pw_name = directory + lcname + '_analysis' + f'/pw_tic{TIC}.csv'

    FL_file = f'/Users/dario/PyhtonPackages/FLOSSY/{FLOSSYsubdir}/tic{TIC}_obs{addpart}.csv'
    amigo_file = directory + lcname + '_analysis' + f'/TIC{TIC}{addpart}.amigo'

    pw = pd.read_csv(pw_name)
    pattern = pd.read_csv(FL_file)

    ids = []
    for p in pattern.period:
        dp = np.abs(pw.period - p)
        if np.nanmin(dp) < 1e-6:
            ids.append(np.argmin(dp))

    amigo_input = pw.iloc[ids].copy()
    amigo_input.sort_values('period', inplace=True)

    amigo_input['e_period'] = amigo_input.e_frequency/(amigo_input.frequency ** 2)

    columns = ['period', 'e_period', 'amp', 'e_amp', 'phase', 'e_phase',
               'stopcrit']

    header_keys = ['per', 'e_per', 'ampl', 'e_ampl', 'phase', 'e_phase',
                   'stopcrit']

    header = '# ' + ' '.join(header_keys)

    np.savetxt(amigo_file, amigo_input[columns].values, header=header,
               delimiter='  ', footer='*'*4, comments='')


def read_amigo(ticid, pattern_path):
    """
    Read the amigo results into a pandas.DataFrame.

    Parameters
    ----------
    ticid : int
        TIC ID.
    pattern_path : str
        Path to the directory with the patterns.

    Returns
    -------
    df : pandas.DataFrame
        AMiGO period-spacing pattern in a DataFrame.

    """
    filename = pattern_path + 'TIC' + str(ticid) + '_spacings.amigo'
    df = pd.read_csv(filename, sep='\t', header=0)
    return df
