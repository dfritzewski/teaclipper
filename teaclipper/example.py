#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:40:28 2025

@author: dario (dario.fritzewski@kuleuven.be)

This script provides an example of the teaclipper workflow.

Note that you should *not* execute this script and hope for a fitted period spacing pattern.
Ideally you use every line individually to create, inspect, and manipulate your light curves.
"""
from . import TUcluster
from . import detrendLCs
from . import LCcheckGUI
from . import prewhitenSTS
from . import findPatterns

TICs = [305906846]
red_name = 'example'

if __name__ == "__main__":
    # get LCs from TESS FFIs
    TUcluster.run_tu(TICs, red_name=red_name)

    # plot light curves and see how much we still have to detrend them
    LCcheckGUI.launch_lc_review_gui(TICs, red_name=red_name, version='TU')

    # additional polynomial detrending
    detrendLCs.all_poly_detrend(TICs, red_name=red_name)

    # Check LCs again
    LCcheckGUI.launch_lc_review_gui(TICs, red_name=red_name)

    # pre-whiten frequencies with Star Shadow
    prewhitenSTS.prewhiten_all(TICs, red_name=red_name)

    # Interactive AMiGO fitting
    findPatterns.analyse_all(TICs, red_name=red_name)
