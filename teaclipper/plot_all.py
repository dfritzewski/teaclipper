#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:56:38 2025

@author: dario (dario.fritzewski@kuleuven.be)

General overview plotting function for TESS lightcurves. It produces a pdf file with all LCs,
periodogram and frequency spectrum.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import lightkurve as lk
from . import utils


def plot_all(df, pdfpath='Overview_all_detrend.pdf', lcpath='LCpathDetrend'):

    df.sort_values('TIC', inplace=True)

    with PdfPages(pdfpath) as pdf:
        for _, row in df.iterrows():

            try:
                if row[lcpath][-3:] == 'csv':
                    lc = lk.LightCurve.read(row[lcpath])
                else:
                    lcdata = np.genfromtxt(row[lcpath]).T
                    lcdf = pd.read_csv(row.LCpath)
                    lcdf.loc[lcdf.time.isin(lcdata[0]), 'flux_detrend'] = lcdata[1]
                    lcdf.loc[lcdf.time.isin(lcdata[0]), 'flux_err_detrend'] = lcdata[2]
                    lc = lk.LightCurve(time=lcdf.time, flux=lcdf.flux_detrend,
                                       flux_err=lcdf.flux_err_detrend)
            except OSError:
                continue
            pg = lc.to_periodogram()

            plot_single(lc, pg, row)

            pdf.savefig()
            plt.close()


def plot_single(lc, pg, row, ret=False):

    sector_borders = get_sectors(lc)
    if type(sector_borders) is int:
        nSec = 1
    else:
        nSec = len(sector_borders) + 1

    fig, ax = make_fig(nSec)

    s_time = np.split(lc.time.value, sector_borders)
    s_flux = np.split(lc.flux.value, sector_borders)

    if nSec == 1:
        plot_LC(s_time[0], s_flux[0], ax[0])

    else:
        for i, time in enumerate(s_time):
            plot_LC(time, s_flux[i], ax[i])

    plot_PG(pg, ax[-2:])

    fig.suptitle(f'TIC {row.TIC}')

    plt.tight_layout()

    if ret:
        return ax


def make_fig(nSec, add=0):
    fig, ax = plt.subplots(nSec + 2 + add, 1, squeeze=True,
                           figsize=(7, 1.4 * (nSec + 2 + add)))

    return fig, ax


def get_sectors(lc):

    timediffs = np.diff(lc.time.value)
    jumps = np.where(timediffs > 1)[0]

    if len(jumps) == 0:
        return 1
    else:
        return jumps + 1


def plot_LC(time, flux, ax):
    ax.plot(time, flux, c='k', lw=0.5)
    ax.set_xlabel('time [MBJD]')
    ax.set_ylabel('rel. flux')

    utils.make_ticks(ax)


def plot_PG(pg, ax, freq_only=False):

    if freq_only:
        a = ax
        lw = 0.25
    else:
        a = ax[0]
        lw = 0.5

    a.plot(pg.frequency, pg.power*1000, c='k', lw=lw)
    a.set_xlim(0, None)
    a.set_xlabel('frequency [d$^-1$]')
    a.set_ylabel('amplitude [ppt]')
    utils.make_ticks(a)

    if not freq_only:
        ax[1].plot(1/pg.frequency, pg.power*1000, c='k', lw=0.5)
        ax[1].set_xlim(0.04, 1)
        ax[1].set_xlabel('period [d]')
        ax[1].set_ylabel('amplitude [ppt]')
        utils.make_ticks(ax[1])


def interact_pg(TICs):
    for TIC in TICs:
        basepath = '/Users/dario/Pleiades/TUgoodtargets/stitchedpleiades/'
        lcname = '_corrected_stitched_detrended'
        lcpath = f'{basepath}lc_tic{TIC}{lcname}.csv'
        lc = lk.LightCurve.read(lcpath)

        pg = lc.to_periodogram()

        plt.plot(1/pg.frequency, pg.power*1000, c='k', lw=0.5)
        plt.xlim(0.1, 1)
        plt.xlabel('period [d]')
        plt.ylabel('amplitude [ppt]')
        plt.title(f'TIC {TIC}')
        utils.make_ticks(plt.gca())
        plt.show(block=True)


def interact_N2516(TICs):
    for TIC in TICs:
        basepath = '/Users/dario/Pleiades/data/all_light_curves_to_Dario/'
        lcname = '_light_curves.dat'
        lcpath = f'{basepath}TIC{TIC}{lcname}'

        lcarr = np.genfromtxt(lcpath).T

        lc = lk.LightCurve(time=lcarr[0], flux=lcarr[1])

        pg = lc.to_periodogram()

        lc.plot()

        plt.figure()
        plt.plot(1/pg.frequency, pg.power*1000, c='k', lw=0.5)
        plt.xlim(0.1, 1)
        plt.xlabel('period [d]')
        plt.ylabel('amplitude [ppt]')
        plt.title(f'TIC {TIC}')
        utils.make_ticks(plt.gca())
        plt.show(block=True)
        plt.close('all')
