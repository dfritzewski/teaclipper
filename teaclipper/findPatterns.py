#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 13:47:25 2025

@author: dario (dario.fritzewski@kuleuven.be)
"""

from . import interactiveAmigo
from matplotlib import pyplot as plt


def analyse_all(TICs, LCdir='./TU/', red_name='', version='detrended', STSdir='./STS/',
                analysis_file=2, pattern_path='./AMiGO_patterns/'):
    '''
    Loop through all stars given in the TICs list to find and fit period spacing patterns
    with AMiGO.

    Parameters
    ----------
    TICs : list-like
        All TIC IDs to consider.
    LCdir : str, optional
        Path to the parent directory of your TESSutils reduction. The default is './TU/'.
    red_name : str, optional
        The name of your TU reduction trailing every directory in your LCdir. The default is ''.
    version : str, optional
        Which version of the light curve to read. Either 'TU' meaning the raw LC as produced
        by TESSutils, or 'detrended' meaining the LC after polynomial detrending. The default
        is 'TU'.
    STSdir : str, optional
        Path to the star shadow put. The code assumes the same filename structure as the
        lightcurves. The default is './STS/'.
    analysis_file : int, optional
        Highest executed step in the star shadow analysis. Set to 2 for only pre-whitening,
        or to 5 for the full eclipsing binary chain. The default is 2.
    pattern_path : str, optional
        Path to directory that (will) store the AMiGO results.
        The default is './AMiGO_patterns/'.

    Returns
    -------
    None.

    '''
    # load the GUI
    amigo_gui = interactiveAmigo.InteractivePatternPlotter(pattern_path=pattern_path)

    # read existing pass/fail lists
    if pattern_path[-1] != '/':
        pattern_path += '/'
    try:
        with open(pattern_path + 'fail_list.txt', 'r', encoding='utf-8') as flist:
            failed_TICs = [int(line.rstrip()) for line in flist]
    except FileNotFoundError:
        failed_TICs = []

    try:
        with open(pattern_path + 'pass_list.txt', 'r', encoding='utf-8') as plist:
            passed_TICs = [int(line.rstrip()) for line in plist]
    except FileNotFoundError:
        passed_TICs = []

    # loop through all TICs
    for TIC in TICs:

        if (TIC in passed_TICs) or (TIC in failed_TICs):
            print(f'Skipping {TIC} as it has been analysed before.')
            continue

        try:
            star = interactiveAmigo.StarData(TIC, LCdir=LCdir, red_name=red_name, STSdir=STSdir,
                                             version=version, analysis_file=analysis_file)
        except FileNotFoundError:
            if analysis_file == 5:
                try:
                    star = interactiveAmigo.StarData(TIC, LCdir=LCdir, red_name=red_name,
                                                     STSdir=STSdir, version=version,
                                                     analysis_file=2)
                except FileNotFoundError:
                    print(f'TIC {TIC} could not be found. Skipping.')
                    continue
                except Exception as e:
                    print(f"Error processing TIC {TIC}: {e}")
                    continue
            else:
                print(f'TIC {TIC} could not be found. Skipping.')
                continue
        except Exception as e:
            print(f"Error processing TIC {TIC}: {e}")
            continue
        # call the analysis
        amigo_gui = interactiveAmigo.InteractivePatternPlotter(pattern_path=pattern_path)
        try:
            amigo_gui.make_analysis(star)
        finally:
            plt.close(amigo_gui.fig)
