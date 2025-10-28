#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:44:20 2023

@author: dario (dario.fritzewski@kuleuven.be)
"""
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from astropy import units as u
import lightkurve as lk
import tessutils as tu


def run_tu(TIC, LCdir='./TU/', red_name="", ncores=8, download=True,
           overwrite=False, sectors=None, binlc=False, time_bin_size=30 * u.min,
           delta_mag=4.5, aperture_mask_threshold=8,
           contamination_level=0.001, aperture_type='auto', extract=True,
           aperture_mask_max_elongation=30, excluded_intervals=None,
           imsize=25, forced_size=[2, 2]):
    '''
    Entry point to reduce a large number of TESS FFI for use in asteroseismology.
    This script can take many TIC IDs and produce its light curves in an automated way.
    Make sure to check the light curves before proceeding.

    Parameters
    ----------
    TIC : int or list-like.
        TIC IDs of the targets of interest.
    LCdir : str, optional
        The directory to write all the output to. Note that during the reduction several
        sub-directories will be created. The default is './', i.e. the current directory.
    red_name : str, optional
        A name to add to the default directory names to distinguish different reductions
        or targets. The default is "".
    ncores : int, optional
        Number of cores to use while processing the data. The default is 8.
    download : bool, optional
        Download the FFI. Useful to be set to False, when doing a re-reduction.
        The default is True.
    overwrite : bool, optional
        Overwrite previous results. If files already exist and False this will stop the
        reduction. The default is False.
    sectors : None or list-like, optional
        Either list of TESS sectors to use or None to use all available sectors.
        The default is None.
    binlc : bool, optional
        Bin the light curve into equal cadence across all sectors. The default is False.
    time_bin_size : astropy.units.quantity.Quantity, optional
        Cadence to use when binning. Use from astropy import units as u.
        The default is 30 * u.min.
    delta_mag : float, optional
        Maximum magnitude difference to consider when estimating the contamination fraction.
        The default is 4.5.
    aperture_mask_threshold : float, optional
        A value for the number of sigma by which a pixel needs to be
        brighter than the median flux to be included in the aperture mask. The default is 8.
    contamination_level : float, optional
        Accepted level of contamination. The default is 0.001.
    aperture_type : str, optional
        Can either be 'auto', 'size', or 'forced'

        auto: Automatic aperture selection. Typical use case.

        size: Define a size of your aperture in forced_size. Used when in a crowded field
        and you know the risk of including contaminating flux.

        forced: Single pixel aperture centred on the star. For very crowded fields to get at
        least a light curve.
        **Use with caution!**

        The default is 'auto'.
    extract : bool, optional
        Do the extraction or not. Use False if you only want to download the FFI cutouts.
        The default is True.
    aperture_mask_max_elongation : int, optional
        Triggers when a 2D aperture mask has 4 or less columns (rows) and at
        least one row (column) larger that `aperture_mask_max_elongation` pixels. In such
        a case, the aperture mask in considered corrupted and the light
        curve extraction is halted. The default is 30.
    excluded_intervals : dict, optional
        Dictionary indicating the intervals to exclude. The keys are
        integers indicating the TESS sectors, and the values are lists of
        tuples, each tuple containing an initial and final time to exclude.
        Times must be given with astropy units.
        Example: dictionary to set to FALSE the quality mask in TESS sectors
        1 and 6 for the given intervals.
            > import astropy.units as u
            > intervals = {}
            > intervals[1] = [ (1334.8, 1335.1)*u.day,
                               (1347.0, 1349.5)*u.day ]
            > intervals[6] = [ (1476.0, 1479.0)*u.day ]
        The default is None.
    imsize : int, optional
        Size in pixels of the FFI cutout. The default is 25.
    forced_size : list, optional
        X and Y dimensions of the forced aperture. Positioning is handled by the pipeline.
        The default is [2, 2].

    Returns
    -------
    None.

    '''

    if LCdir[-1] != '/':
        LCdir += "/"

    download_dir = f"{LCdir}tpf{red_name}/"

    if download:
        tu.reduction.download_tpf(TIC,
                                  ncores=1,
                                  imsize=imsize,
                                  outputdir=download_dir,
                                  overwrite=overwrite,
                                  sectors=sectors)

    # get list of tpf files
    if np.ndim(TIC) != 0:
        TPF_files = []
        for T in TIC:
            TPF_files += [str(f) for f
                          in Path(download_dir).glob(f'tic{T}_sec*.fits')]
    else:
        TPF_files = [str(f) for f
                     in Path(download_dir).glob(f'tic{TIC}_sec*.fits')]

    lc_dir = f"{LCdir}processed{red_name}"

    # set the number of pc bins and threshold value according to the time resolution
    # TODO: this is just a tempory work around, should be implemented in TU properly
    if (sectors is None) or (sectors[0] > 55):
        num_of_pc_bins = 120
        pc_th_variance = 2.5e-6
    else:
        num_of_pc_bins = 40
        pc_th_variance = 1e-5

    if extract:
        tu.reduction.extract_light_curve(np.array(TPF_files),
                                         overwrite=overwrite,
                                         delta_mag=delta_mag,
                                         ncores=ncores,
                                         outputdir=lc_dir,
                                         aperture_mask_threshold=aperture_mask_threshold,
                                         aperture_mask_min_pixels=1,
                                         max_num_of_pc=7,
                                         aperture_mask_max_elongation=aperture_mask_max_elongation,
                                         contamination_level=contamination_level,
                                         num_of_pc_bins=num_of_pc_bins,
                                         pc_threshold_variance=pc_th_variance,
                                         aperture_type=aperture_type,
                                         forced_size=forced_size,
                                         excluded_intervals=excluded_intervals)

    if sectors is None:
        sectors = "all"

    # group LCs
    red_dir = f"{LCdir}grouped{red_name}"
    tu.reduction.group_lcs(lc_dir, outputdir=red_dir, TICs=TIC, sectors=sectors)

    # stitch LCs
    sticht_dir = f"{LCdir}stitched{red_name}"
    tu.reduction.stitch_group(red_dir, TICs=TIC, outputdir=sticht_dir,
                              overwrite=overwrite, binlc=binlc,
                              time_bin_size=time_bin_size)

    # plot diagnosis
    pdfdir = f"{LCdir}diagnosis{red_name}"
    Path(pdfdir).mkdir(parents=True, exist_ok=True)

    if np.ndim(TIC) != 0:
        for T in TIC:
            stich_plot(T, sticht_dir, red_dir, binlc, pdfdir)
    else:
        stich_plot(TIC, sticht_dir, red_dir, binlc, pdfdir)

    # write log file with parameters for future reference
    with open(f'{LCdir}{red_name}.log', 'w', encoding="utf-8") as lf:
        now = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
        lf.write(f'Log for TU run {red_name} on {now}\n')
        lf.write(f'Data saved to {LCdir}\n\n')
        lf.write('List of TICs \n')
        if np.ndim(TIC) != 0:
            lf.write("\n".join(str(T) for T in TIC))
            lf.write('\n')
        else:
            lf.write(f'{TIC}\n')

        lf.write('\n\n')
        lf.write('The follwing parameters were use:\n')
        lf.write(f'ncores={ncores}\n')
        lf.write(f'download={download}\n')
        lf.write(f'overwrite={overwrite}\n')
        lf.write(f'sectors={sectors}\n')
        lf.write(f'binlc={binlc}\n')
        lf.write(f'delta_mag={delta_mag}\n')
        lf.write(f'time_bin_size={time_bin_size}\n')
        lf.write(f'aperture_mask_threshold={aperture_mask_threshold}\n')
        lf.write(f'contamination_level={contamination_level}\n')
        lf.write(f'faperture_type={aperture_type}\n')
        lf.write(f'forced_size={forced_size}\n')
        lf.write(f'extract={extract}\n')
        lf.write(f'aperture_mask_max_elongation={aperture_mask_max_elongation}\n')
        lf.write(f'excluded_intervals={excluded_intervals}\n')
        lf.write(f'imsize={imsize}\n')

        lf.write('\nTU cluster uses these additional default parameters:\n')
        lf.write('aperture_mask_min_pixels=1, max_num_of_pc=7\n')

        lf.close()
    print('Done.')


def stich_plot(TIC, inputdir, group_dir, binlc, savedir):
    '''
    Create the diagnostic plots.

    Parameters
    ----------
    TIC : int
        TIC ID of the star.
    inputdir : str
        Path to the directory with the stitched light curves.
    group_dir : str
        Path to the directory with the grouped light curve.
    binlc : bool
        Bin the light curve into equal cadence across all sectors.
    savedir : str
        Path to the directory for the diagnosis files.

    Returns
    -------
    None.

    '''

    try:
        lc = pd.read_csv(f'{inputdir}/lc_tic{TIC}_corrected_stitched.csv')
    except FileNotFoundError:
        return

    lc = lk.LightCurve(time=lc.time, flux=lc.flux)

    group_file = f'{group_dir}/tic{TIC}_allsectors_corrected.pickle'
    pdfname = f'{savedir}/diagnostic_plot_TIC{TIC}.pdf'
    tu.plots.plot_diagnosis(group_file, binlc=binlc, pdfname=pdfname, rasterize_lc=True)


def read_plot_interactive(TIC, basedir='./', outdirext=""):
    '''
    Loads the light curve to display it and a periodogram for manual inspection.
    E.g. To find the times stretches that are heavily contaminated.

    Parameters
    ----------
    TIC : TYPE
        DESCRIPTION.
    basedir : TYPE, optional
        DESCRIPTION. The default is './'.
    outdirext : TYPE, optional
        DESCRIPTION. The default is "".

    Returns
    -------
    None.

    '''
    try:
        lc = pd.read_csv(f'{basedir}stitched{outdirext}/lc_tic{TIC}_corrected_stitched.csv')
    except FileNotFoundError:
        return

    lc = lk.LightCurve(time=lc.time, flux=lc.flux)
    pg = lc.to_periodogram()

    lc.plot()
    pg.plot()
