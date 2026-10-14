#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 13:44:31 2025

@author: dario (dario.fritzewski@kuleuven.be)

Adjusted code from Alex Kemp.
"""
import warnings
import os
import copy
import ast
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import chi2 as chi2dist
import astropy.units as u

import amigo
from amigo.observations import pulsations as obspuls
from amigo.asymptotic_theory import gravity_modes
from amigo import fit
from amigo.periodogram import scargle
from amigo import compute_rotation
import star_shadow as sts

from . import utils


class AMiGO_data:
    """
    This class holds the AMiGO data associated with the fit of the g-mode pattern.
    """

    def __init__(self, TIC, res_path='./AMiGO_patterns/'):
        """
        Init AMiGO data holder.

        Parameters
        ----------
        TIC : int
            The TIC ID of the star.
        res_path : str, optional
            Path of the parent directory holding the AMiGO pattern / to which the results will be
            saved. The default is './AMiGO_patterns/'.

        Returns
        -------
        None.

        """
        # basis info
        self.TIC = TIC
        self.name = f'TIC{TIC}'
        self.pattern_name = f'TIC{TIC}_spacings.amigo'

        # make path
        if res_path[-1] != '/':
            res_path += '/'
        self.res_path = res_path

        self.pattern_path = self.res_path + self.pattern_name

        # Try reading the pattern if no obsstar object is provided
        try:
            self.obsstar = obspuls(self.name, self.pattern_path)
            print(f'Read existing AMiGO file at {self.pattern_path}')
        except FileNotFoundError:
            self.obsstar = None
            print('Did not find existing AMiGO results. Initializing empty object.')

        # extract all info if obstar exists
        if self.obsstar is not None:
            self.split_periods = self.obsstar.period(split=True)
            self.split_ampls = self.obsstar.amplitude(split=True)
            self.pers, self.e_pers, self.dps, self.e_dps, self.solid, \
                self.dashed = self.obsstar.patterns(to_plot=True)

        else:
            self.split_periods = []
            self.split_ampls = []
            self.pers = []
            self.e_pers = []
            self.dps = []
            self.e_dps = []
            self.solid = []
            self.dashed = []

        # try reading previous results
        try:
            self.get_amigo_results()
        except FileNotFoundError:
            self.amigo_res_df = None
            self.Pi0 = None
            self.Pi0_e = None
            self.frot = None
            self.frot_e = None
            self.alpha = None
            self.chi_sq = None
            self.gyre_dir = None
            self.modes = None
            self.amigo_kwarg = None

        # init parameters
        self.deg_freedom = None
        self.sample = None
        self.chi2_full = None

    def get_amigo_results(self):
        '''
        Read AMiGO results.

        Returns
        -------
        None.

        '''
        self.amigo_res_df = pd.read_csv(self.res_path+self.name+'_modelling_summary.csv')
        # assign values for easier access
        self.Pi0 = self.amigo_res_df.Pi0.values[0]
        self.Pi0_e = self.amigo_res_df.Pi0_e.values[0]
        self.frot = self.amigo_res_df.frot.values[0]
        self.frot_e = self.amigo_res_df.frot_e.values[0]
        self.alpha = self.amigo_res_df.alpha.values[0]
        self.chi_sq = self.amigo_res_df.chi2_min.values[0]
        self.gyre_dir = self.amigo_res_df.chi2_min.values[0]
        self.amigo_kwarg = self.amigo_res_df.amigo_command.values[0]

        try:
            amigo_kwarg = ast.literal_eval(self.amigo_kwarg)
            self.modes = [gravity_modes(self.gyre_dir, kval=amigo_kwarg['kval'],
                                        mval=amigo_kwarg['mval'])]
        except Exception:
            self.modes = None


class StarData:
    '''
    This class holds the observational data. It reads the light curve and prewhitening resusults.
    '''

    def __init__(self, TIC, LCdir='./TU/', red_name='', version='detrended',
                 STSdir='./STS/', analysis_file=2):
        """
        Init for the class that holds the observational data.

        *The paths and read_lc are made for the data as obtained with tessutils and starshadow.*
        If you want to read some other lightcurves adjust LCdir and self.lcpath. Similarly you have
        to adjust self.pwpath. If your light curves are not stored in csv files load_lc needs also
        be adjusted.

        Parameters
        ----------
        TIC : int
            TIC ID of the target.
        LCdir : str, optional
            Path to directory that holds all lightcurves. (The tessutils main directory of the
                                                           reduction). The default is './TU/'.
        red_name : str, optional
            The name of your TU reduction trailing every directory in your LCdir. The default is ''.
        version : str, optional
            Which version of the light curve to read. Either 'TU' meaning the raw LC as produced
            by TESSutils, or 'detrended' meaining the LC after polynomial detrending. The default
            is 'detrended'.
        STSdir : str, optional
            Path to the star shadow put. The code assumes the same filename structure as the
            lightcurves. The default is './STS/'.
        analysis_file : int, optional
            Highest executed step in the star shadow analysis. Set to 2 for only pre-whitening,
            or to 5 for the full eclipsing binary chain. The default is 5.

        Returns
        -------
        None.

        """
        # basic info about LC
        self.TIC = TIC
        self.name = f'TIC{TIC}'
        self.LCdir = LCdir
        self.red_name = red_name
        self.version = version
        self.analysis_file = analysis_file

        # build LC path from given names
        if version == 'TU':
            lcname = '_corrected_stitched'
        elif version == 'detrended':
            lcname = '_corrected_stitched_detrended'
        else:
            lcname = ''

        LCdir = utils.make_dir_path(LCdir, red_name)

        self.lcpath = f'{LCdir}lc_tic{TIC}{lcname}.csv'

        lc_name_sts = f'lc_tic{TIC}_corrected_stitched'
        self.pwpath = (STSdir + f'{self.red_name}/' + lc_name_sts
                       + '_analysis/' + f'{lc_name_sts}_analysis_{self.analysis_file}.hdf5')

        # load LC and STS results
        self.load_lc()
        self.load_pw()

    def load_lc(self):
        '''
        Loading function for light curve.

        Returns
        -------
        None.

        '''
        lcdf = pd.read_csv(self.lcpath)
        self.lc = lcdf

    def load_pw(self):
        '''
        Loading function for star shadow pre-whitening results.

        Returns
        -------
        None.

        '''
        print(f'reading STS results from: {self.pwpath}')
        sts_dict = sts.ut.read_parameters_hdf5(self.pwpath, h5py_file_kwargs={'locking': False})
        self.sts_res = sts_dict


class InteractivePatternPlotter:
    '''
    Main class that enables the interactive AMiGO period-spacing patterns.
    stardata = StarData(TIC)
    Usage InteractivePatternPlotter(pattern_path='./AMiGO_patterns/').make_analysis(stardata)
    '''

    def __init__(self,
                 pattern_path='./AMiGO_patterns/',
                 fail_list_name='fail_list.txt',
                 softfail_list_name='softfail_list.txt',
                 pass_list_name='pass_list.txt'
                 ):
        '''
        Creation of plot and setting important parameters. Not all final contents of self are
        initialized here.

        Parameters
        ----------
        pattern_path : str, optional
            Path to directory that (will) store the AMiGO results.
            The default is './AMiGO_patterns/'.
        fail_list_name : str, optional
            Full path to a file that holds the TIC IDs of all stars that did not pass the PSP
            search. The default is 'fail_list.txt'.
        softfail_list_name : str, optional
            Full path to a file that holds the TIC IDs of all stars that you might want to
            reconsider later. The default is 'softfail_list.txt'.
        pass_list_name : str, optional
            Full path to a file that holds the TIC IDs of all stars with successful AMiGO fits.
            The default is 'pass_list.txt'.

        Returns
        -------
        None.

        '''
        # set paths
        if pattern_path[-1] != '/':
            pattern_path += '/'
        self.pattern_path = pattern_path

        Path(self.pattern_path).mkdir(parents=True, exist_ok=True)

        self.fail_list_name = fail_list_name
        self.softfail_list_name = softfail_list_name
        self.pass_list_name = pass_list_name

        # init lists for picking periods/gaps
        self.indpicklist = []
        self.picklistx = []
        self.picklisty = []
        self.gaplist = []

        # init figure and axis
        self._setup_plot()
        self._connect_events()

        # set status of fit
        self.fitted = False

    def _setup_plot(self):
        '''
        Set-up routine for all the different panels of the main window.

        Returns
        -------
        None.

        '''
        self.fig = plt.figure(figsize=(15, 10))

        # Define the main grid (4 rows high, 3 columns wide)
        gs = GridSpec(4, 3, figure=self.fig, height_ratios=[5.5, 1, 3, 2],
                      width_ratios=[4.5, 2.5, 1])

        # Adjust spacing between subplots
        gs.update(left=0.08, right=0.98, bottom=0.08, top=0.85,
                  wspace=0.3, hspace=0.3)

        # 1. Periodogram Axis (ax_pg)
        self.ax_pg = self.fig.add_subplot(gs[0:1, :])
        utils.make_ticks(self.ax_pg)

        # 2. Pattern Axis (ax_psp)
        self.ax_psp = self.fig.add_subplot(gs[2:, 0])
        utils.make_ticks(self.ax_psp)

        # 3. Chi-squared Landscape Axis (ax_chi)
        self.ax_chi = self.fig.add_subplot(gs[2:, 1])
        utils.make_ticks(self.ax_chi)

        # 4. 1D Chi-squared Axes (ax_frot, ax_pi0)
        gs_1d_chi = GridSpecFromSubplotSpec(3, 1, subplot_spec=gs[2:4, 2], height_ratios=[1, 4, 4],
                                            hspace=0.05)

        self.ax_frot = self.fig.add_subplot(gs_1d_chi[1, 0])

        self.ax_pi0 = self.fig.add_subplot(gs_1d_chi[2, 0])
        utils.make_ticks(self.ax_pi0)

        # Configure frot axis
        self.ax_frot.tick_params(labelbottom=False, labeltop=True)  # Hide x-labels on frot
        utils.make_ticks(self.ax_frot)

        # Set labels and titles
        self.ax_pg.set_xlabel('Period [d]')
        self.ax_pg.set_ylabel('Amplitude')

        self.ax_psp.set_xlabel('Period [d]')
        self.ax_psp.set_ylabel(r'$\Delta P$ [s]')

    def _connect_events(self):
        '''
        Routine to connect to the different possible interaction events.

        Returns
        -------
        None.

        '''
        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('button_press_event', self._on_right_click)

    def _clear_ax(self, ax):
        '''
        Clear axis to redraw plot.

        Parameters
        ----------
        ax : mpl.axis
            The axis to be cleared.

        Returns
        -------
        None.

        '''
        for artist in ax.lines + ax.collections:
            artist.remove()

    def _update_plot(self):
        '''
        Redraw all plots that needs to be updated after an interaction.

        Parameters
        ----------
        plot_pattern : bool, optional
            Indicates whether the period spacing pattern should also be updated.
            The default is True.

        Returns
        -------
        None.

        '''
        # get old limits
        ax_xlim = self.ax_pg.get_xlim()
        ax_ylim = self.ax_pg.get_ylim()

        self._clear_ax(self.ax_pg)
        self._clear_ax(self.ax_psp)

        # redraw all plots
        self._make_title()
        self._plot_periodogram()

        self._plot_pattern()

        if (self.amigo_res.modes is not None) and (self.fitted):
            self._plot_theory_pattern_live()

        # reset limits and finalize
        self.ax_pg.set_xlim(ax_xlim)
        self.ax_pg.set_ylim(ax_ylim)
        self.ax_psp.set_xlim(ax_xlim)
        self.fig.canvas.draw()

    def _plot_periodogram(self):
        '''
        Main plotting function for the periodogram.

        Returns
        -------
        None.

        '''
        # unpack periodogram
        scgfreq_residharm, scgamp_residharm = self.pg_residharm
        scgfreq_residall, scgamp_residall = self.pg_residall

        # plot pg and residuals after pre-whitening
        self.ax_pg.plot(1 / scgfreq_residharm, scgamp_residharm, color='lightgrey',
                        label="LS resid_harm")
        self.ax_pg.plot(1 / scgfreq_residall, scgamp_residall, color='grey', label="LS resid_all")

        # add pre-whitened frequencies
        self.ax_pg.vlines(1 / self.good_freq, np.zeros_like(self.good_freq), self.good_amp,
                          colors='mediumaquamarine',
                          label="pass_snr_sigma")
        self.ax_pg.vlines(1 / self.freq[self.fail_bool], np.zeros_like(self.freq[self.fail_bool]),
                          self.amp[self.fail_bool], colors='b', label="fail_snr_sigma", zorder=1)

        # add orbital harmonics of eclipsing binary
        if self.star_data.analysis_file > 3 and self.h_bool.any():
            self.ax_pg.vlines(self.harm_pred_P, 0, max(self.amp[self.h_bool]), colors='wheat',
                              linestyles=':', label='Orbital harmonics')
            self.ax_pg.vlines(1 / self.freq[self.h_bool], np.zeros_like(self.freq[self.h_bool]),
                              self.amp[self.h_bool], colors='orange', linestyles=':',
                              label='Orbital harmonics')

        # add markers for pre-whitened frequencies
        self.ax_pg.scatter(1 / self.good_freq, self.good_amp, c=self.snr[~self.h_bool], marker='x',
                           label='S/N', vmin=0, vmax=self.snr_thresh, cmap='bwr', zorder=2,
                           picker=True, pickradius=5)

        # overplot selected frequencies and gaps
        self.ax_pg.scatter(self.picklistx, self.picklisty, c='g', marker='x', zorder=3, picker=True)
        self.ax_pg.vlines(self.gaplist, 0, max(self.amp), colors='g')

    def _plot_lc(self):
        '''
        Plot light curve. Only used in initial step before the first AMiGO run.

        Returns
        -------
        None.

        '''
        self.ax_chi.plot(self.star_data.lc.time.values, self.star_data.lc.flux.values, c='k')
        # somehow in the example data this is offset by one of you notice
        # such an offset remove the +1 below.
        self.ax_chi.plot(self.star_data.lc.time.values, self.flux_harm_resid_lin + 1, c='r')

        self.ax_chi.set_xlabel('BJD - 2457000 [d]')
        self.ax_chi.set_ylabel('relative flux')

    def _plot_pattern(self):
        '''
        Plot the period spacing pattern.

        Returns
        -------
        None.

        '''
        try:
            structured_pointlist = self._makegaps()
            for sublist in structured_pointlist:
                self.ax_psp.plot(sublist[:-1], np.diff(sublist) * 24 * 3600, marker='.',
                                 color='g', linewidth=5)
        except Exception as error:  # which error can occur here?
            print(error)

    def _prepare_data(self):
        '''
        Prepare all the data and create the periodograms. Many new variables are added to self here.

        Returns
        -------
        None.

        '''
        # get lists of selected periods and gaps
        if self.amigo_res.obsstar is not None:
            self.picklistx = list(self.amigo_res.split_periods[0].value)
            self.picklisty = list(self.amigo_res.split_ampls[0])
            self._create_gap_list()

        # extract all data needed for plotting
        # time stamps of LC
        t = self.star_data.lc.time.values

        # pre-whitened frequencies, etc.
        sin_mean = self.star_data.sts_res['sin_mean']
        self.freq = sin_mean[2]
        self.amp = sin_mean[3]
        self.phase = sin_mean[4]

        # masks of pre-whitening results
        sin_select = self.star_data.sts_res['sin_select']
        self.b_bool = sin_select[2]  # passed both sigma and snr boolean
        self.h_bool = sin_select[3]  # harmonic boolean
        self.fail_bool = ~self.h_bool * ~self.b_bool  # in significant frequencies

        self.good_freq = self.freq[~self.h_bool]
        self.good_amp = self.amp[~self.h_bool]
        self.good_ph = self.phase[~self.h_bool]

        # model LC based on extracted quantities
        model_sines = sts.tsf.sum_sines(t, self.freq, self.amp, self.phase)  # fullmodel
        model_sines_harm = sts.tsf.sum_sines(t, self.freq[self.h_bool], self.amp[self.h_bool],
                                             self.phase[self.h_bool])  # binary harmonics
        model_line = sts.tsf.linear_curve(t, sin_mean[0], sin_mean[1],
                                          self.star_data.sts_res['i_sectors'])  # linear trend

        # light curves with transits removed
        self.flux_harm_resid_lin = self.star_data.lc.flux.values - model_sines_harm - model_line

        # periodograms
        self.pg_full = scargle(t, self.star_data.lc.flux.values - model_line)  # full data set
        self.pg_residharm = scargle(t, self.flux_harm_resid_lin)  # non-harmonics, i.e. oscillations
        self.pg_residall = scargle(t, self.star_data.lc.flux.values - model_sines
                                   - model_line)  # all residuals

        # get noise and snr threshold
        noise = self._calc_noise(self.freq, self.pg_residall[0], self.pg_residall[1])
        self.snr = self.amp / noise
        self.snr_thresh = sts.ut.signal_to_noise_threshold(len(self.star_data.lc.time))

        sin_select = self.star_data.sts_res['sin_select']
        self.b_bool = sin_select[2]                      # passed both sigma and S/N

        self.h_bool = np.zeros(len(self.freq), dtype=bool)
        # additional data for eclipsing binaries
        if self.star_data.analysis_file > 3:
            p_orb = self.star_data.sts_res['ephem'][0]
            harm_pred = np.arange(1, int(self.pg_full[0].max() * p_orb) + 1) / p_orb
            self.harm_pred_P = 1 / harm_pred
        else:
            self.harm_pred_P = None

        if len(self.picklistx) > 0:
            # for every picked period find the closest macthing pre-whitened frequency
            for i, per in enumerate(self.picklistx):
                ind = np.argmin(np.abs(per - 1 / self.good_freq))
                self.picklistx[i] = (1 / self.good_freq)[ind]
                self.picklisty[i] = self.good_amp[ind]

                self.indpicklist.append(ind)

    def _plot_theory_pattern_live(self):
        '''
        Plot the asymptotic pattern as returned by AMiGO.

        Returns
        -------
        None.

        '''
        # plotting the asymptotic models
        # print(self.amigo_res.alpha, self.amigo_res.modes)
        for ialpha, mode in zip([self.amigo_res.alpha], self.amigo_res.modes):
            pattern = mode.uniform_pattern(self.amigo_res.frot / u.d, self.amigo_res.Pi0 * u.s,
                                           alpha_g=ialpha, unit='days')
            # print(pattern)
            # dealing with r-modes...
            if mode.kval < 0:
                pattern = -pattern[::-1]
            line_length = np.ones(len(pattern)) * np.amax(self.amigo_res.obsstar.amplitude()) * 1.1
            self.ax_pg.vlines(pattern, np.zeros(len(pattern)),
                              line_length, colors='r')
            self.ax_pg.plot(pattern[:-1], np.diff(pattern.to(u.s)), 'r-')

        # plotting the observations
        for per, ampl in zip(self.amigo_res.split_periods, self.amigo_res.split_ampls):
            self.ax_pg.vlines(per, np.zeros(len(per)), ampl, colors='k', lw=1.5)

        spmin = 10000000. * u.s
        spmax = 0. * u.s

        # plotting the fittted modes
        per, eper, dp, edp, line, dash = self.amigo_res.obsstar.patterns(to_plot=True)

        for sl in line:
            plot_line = LineCollection(sl, linewidths=1., linestyles='-', colors='k')
            self.ax_psp.add_collection(plot_line)
        for dl in dash:
            plot_dash = LineCollection(dl, linewidths=1., linestyles=':', colors='k')
            self.ax_psp.add_collection(plot_dash)
        # print(per, [dper.to(u.s) for dper in dp], eper, [ed.to(u.s) for ed in edp])

        # adding uncertainties
        for i, p in enumerate(per):
            self.ax_psp.errorbar(p, dp[i].to(u.s), xerr=eper[i], yerr=edp[i].to(u.s), fmt='k.',
                                 mfc='k', ecolor='k', elinewidth=1.)
            spmin = min(spmin, np.amin(dp[i].to(u.s) - edp[i].to(u.s)))
            spmax = max(spmax, np.amax(dp[i].to(u.s) + edp[i].to(u.s)))

        # spran = spmax - spmin
        # ax_diff.set_ylim(max(0.,spmin.value-0.05*spran.value), spmax.value+0.05*spran.value)
        # ax_patt.set_ylim(0., np.amax(obsstar.amplitude())*1.1)
        # plt.xlim(permin.value, permax.value)
        self.fig.canvas.draw()

    def _plot_chi2_space_live(self):
        '''
        Plot the chi^2 landescape after an AMiGO run.

        Returns
        -------
        None.

        '''
        self._clear_ax(self.ax_frot)
        self._clear_ax(self.ax_pi0)
        self._clear_ax(self.ax_chi)

        # make chisq landscape
        redchi2_cutoff = (chi2dist.ppf(0.999, self.amigo_res.deg_freedom)
                          / self.amigo_res.deg_freedom)

        N_frot = len(np.unique(self.amigo_res.sample['f_rot']))
        N_Pi0 = len(np.unique(self.amigo_res.sample['Pi0']))
        loggrid = np.log10(self.amigo_res.chi2_full.reshape((N_Pi0, N_frot)))

        # plot boundaries
        uni_frot = np.unique(self.amigo_res.sample['f_rot'])
        uni_Pi0 = np.unique(self.amigo_res.sample['Pi0'])
        extent_plot = (np.amin(uni_frot), np.amax(uni_frot), np.amin(uni_Pi0), np.amax(uni_Pi0))

        chi2_plot = np.r_[self.amigo_res.chi2_full <=
                          np.amin(self.amigo_res.chi2_full) * redchi2_cutoff]

        # plot samples
        self.ax_frot.semilogy(self.amigo_res.sample['f_rot'][chi2_plot],
                              self.amigo_res.chi2_full[chi2_plot], 'k.')
        self.ax_frot.set_xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        self.ax_frot.xaxis.set_label_position('top')
        self.ax_frot.set_ylabel(r'$\chi^2$')

        self.ax_pi0.semilogy(self.amigo_res.sample['Pi0'][chi2_plot],
                             self.amigo_res.chi2_full[chi2_plot], 'k.')
        # self.ax_pi0.set_xlim(np.nanmin(self.amigo_res.sample['Pi0'][chi2_plot]),
        #                      np.nanmax(self.amigo_res.sample['Pi0'][chi2_plot]))
        self.ax_pi0.set_xlabel(r'$\sf \Pi_{\sf 0}$ (s)')
        self.ax_pi0.set_ylabel(r'$\chi^2$')

        # calculate confidence intervals
        levels = []
        for lev in [0.99, 0.95, 0.6827]:
            levels.append([np.amin(self.amigo_res.chi2_full) *
                           chi2dist.ppf(lev,
                                        self.amigo_res.deg_freedom) / self.amigo_res.deg_freedom])

        # full plot
        _ = self.ax_chi.imshow(loggrid, extent=extent_plot, interpolation='bicubic', origin='lower',
                               aspect="auto", cmap=plt.get_cmap('gray'))
        # 3 sigma
        _ = self.ax_chi.contour(loggrid, np.log10(np.array(levels[0])), colors='r', linewidths=2,
                                linestyles=':', extent=extent_plot)
        # 2 sigma
        _ = self.ax_chi.contour(loggrid, np.log10(np.array(levels[1])), colors='r', linewidths=2,
                                linestyles='--', extent=extent_plot)
        # 1 sigma
        _ = self.ax_chi.contour(loggrid, np.log10(np.array(levels[2])), colors='r', linewidths=2,
                                extent=extent_plot)
        # plot best fitting point
        self.ax_chi.plot([self.amigo_res.frot], [self.amigo_res.Pi0], 'wx')

        # ax limits and labels
        self.ax_chi.set_xlim(np.amin(self.amigo_res.sample['f_rot'][chi2_plot]),
                             np.amax(self.amigo_res.sample['f_rot'][chi2_plot]))
        self.ax_chi.set_ylim(np.amin(self.amigo_res.sample['Pi0'][chi2_plot]),
                             np.amax(self.amigo_res.sample['Pi0'][chi2_plot]))
        self.ax_chi.set_xlabel(r'$\sf f_{\sf rot}$ ($\sf d^{-1}$)')
        self.ax_chi.set_ylabel(r'$\sf \Pi_{\sf 0}$ (s)')

    def _make_title(self):
        '''
        Add the top in the top centre with the info on the star and AMiGO results.

        Returns
        -------
        None.

        '''

        if self.amigo_res.amigo_res_df is not None:
            self.fig.suptitle(f"TIC {self.star_data.TIC}\n" +
                              fr"$\Pi_0=$ {self.amigo_res.Pi0:.0f} $\pm$ {self.amigo_res.Pi0_e:.2f}, " +
                              r"$f_{\rm rot}=$" +
                              fr"{self.amigo_res.frot:.3f} $\pm$ {self.amigo_res.frot_e:.3f}" +
                              "\n" +
                              fr"$\alpha=$ {self.amigo_res.alpha:.3f}, " +
                              r"$\chi^2_{\rm min}=$" + f"{self.amigo_res.chi_sq:.0f}")
        else:
            self.fig.suptitle(f"TIC {self.star_data.TIC}\n")

    def _create_gap_list(self):
        '''
        Recreate the gap list based on the saved AMiGO pattern

        Returns
        -------
        None.

        '''
        seq_id = self.amigo_res.obsstar.sequence  # sequence ids from AMiGO

        gap_id = np.where(np.diff(seq_id) > 1)[0]  # find jumps

        pers = self.amigo_res.split_periods[0].value

        gaps = [(pers[g] + pers[g + 1]) / 2 for g in gap_id]  # put gaps at mean between two parts

        self.gaplist = gaps

    def _makegaps(self):
        '''
        Add the gaps properly to the data.

        Returns
        -------
        list
            List of indivual continous stretched of observed period spacing.

        '''
        sortinds = np.argsort(self.picklistx)
        sorted_picklistx = np.array(self.picklistx)[sortinds]
        self.gaplist.sort()

        outlist = []
        sublist = []
        gap_index = 0

        if not self.gaplist:
            return [sorted_picklistx]

        for point in sorted_picklistx:

            # Check if the current point is smaller than the current gap
            # The while loop handles cases where multiple gaps fall between points
            while gap_index < len(self.gaplist) and point >= self.gaplist[gap_index]:
                # The gap has been crossed. Save the completed sublist.
                outlist.append(sublist)

                sublist = []  # reset

                gap_index += 1

            # Add the point to the current sublist (always happens after gap check)
            sublist.append(point)

        outlist.append(sublist)  # append the final sublist

        # remove empty sublists
        return [sub for sub in outlist if sub]

    def _calc_noise(self, freq, spectra_freq, spectra_amp, boxwidth=1.):
        '''
        Calculate the local noise of the periodogram.

        Parameters
        ----------
        freq : np.array
            Frequencies for which the noise needs to be calculated.
        spectra_freq : np.array
            Frequncies of the periodogram.
        spectra_amp : np.array
            Amplitudes of the periodogram.
        boxwidth : float, optional
            Width in frequency space over which the noise should be calculated. The default is 1.0.

        Returns
        -------
        noise : np.array
            Noise at every frequency.

        '''
        # init box
        boxmin = freq - boxwidth / 2
        boxmax = freq + boxwidth / 2
        boxmin[boxmin < 0] = 0
        noise = []

        # iterate over frequencies
        for i, _ in enumerate(freq):
            boxbool = (boxmin[i] < spectra_freq) & (spectra_freq < boxmax[i])
            noise.append(np.median(spectra_amp[boxbool]))

        return noise

    def _run_and_save_amigo(self, command):
        '''
        The call to AMiGO is happening here. It also write the files needed by AMiGO and retrieves
        the results.

        Parameters
        ----------
        command : str
            A str of a dict with the AMiGO input parameters, i.e. the content of the text box.

        Returns
        -------
        None.

        '''
        sortinds = np.argsort(self.picklistx)
        s_ind = np.array(self.indpicklist)[sortinds]

        # write periods, etc. to amigo file for fitting
        # start with header
        header_cols = ["per", "e_per", "ampl", "e_ampl", "phase", "e_phase", "stopcrit"]

        if self.gaplist:
            sorted_gaplist = copy.copy(self.gaplist)
            sorted_gaplist.sort()
        else:
            sorted_gaplist = self.gaplist

        # for every period get all the values needed by amigo
        per_line = 1 / self.good_freq[s_ind]
        e_per_line = per_line * (self.star_data.sts_res['sin_err'][2][~self.h_bool][s_ind]
                                 / self.good_freq[s_ind])
        ampl_line = self.good_amp[s_ind]
        e_ampl_line = self.star_data.sts_res['sin_err'][3][~self.h_bool][s_ind]
        phase_line = self.good_ph[s_ind]
        e_phase_line = self.star_data.sts_res['sin_err'][4][~self.h_bool][s_ind]
        stopcrit = np.ones_like(per_line)

        # join in one data frame
        df = pd.DataFrame(np.array([per_line, e_per_line, ampl_line, e_ampl_line, phase_line,
                                    e_phase_line, stopcrit]).T,
                          columns=header_cols)

        # handle gaps and insert nan where appropriate
        df['Next_Period'] = df['per'].shift(-1)

        insertion_indices = []  # holding the info on where to add gaps

        for gap_value in self.gaplist:
            # Find the first row 'i' where Period_i <= gap_value < Next_Period_i
            condition = (df['per'] <= gap_value) & (df['Next_Period'] > gap_value)

            # Get the index of the first row that satisfies the condition
            insertion_index = df[condition].index.max()

            if pd.notna(insertion_index):
                insertion_indices.append(insertion_index)

        temp_df = []  # not really a df
        nan_row = pd.Series([np.nan] * len(df.columns), index=df.columns)  # the nan_row

        for i in df.index:
            temp_df.append(df.loc[i])  # Append the original row

            if i in insertion_indices:  # if row in list add nan row
                temp_df.append(nan_row)

        # Clean up temporary column
        final_df = pd.DataFrame(temp_df,
                                columns=df.columns).drop(columns=['Next_Period'],
                                                         errors='ignore').reset_index(drop=True)
        Path(self.amigo_res.res_path).mkdir(parents=True, exist_ok=True)

        # write to file
        with open(self.amigo_res.pattern_path, 'w+', encoding="utf-8") as f:
            f.write('# ' + "           ".join(header_cols) + '\n')  # write AMiGO header

            # write data
            final_df.to_csv(f, sep='\t', index=False, header=False, float_format='%.16f',
                            na_rep='--')
            f.write("*" * 111 + "\n \n")  # write final line

        # call amigo and fit
        amigo_kwargs = ast.literal_eval(command)
        print(amigo_kwargs)
        self.predict_pattern(**amigo_kwargs)

        # write out a file with the results and command
        summary_dict = {'frot': [self.amigo_res.frot],
                        'frot_e': [self.amigo_res.frot_e],
                        'Pi0': [self.amigo_res.Pi0],
                        'Pi0_e': [self.amigo_res.Pi0_e],
                        'alpha': [self.amigo_res.alpha],
                        'chi2_min': [self.amigo_res.chi_sq],
                        'amigo_command': [command],
                        'gyre_dir': [self.amigo_res.gyre_dir]}
        summary_df = pd.DataFrame.from_dict(summary_dict, orient='columns')

        summary_df.to_csv(f"{self.pattern_path}TIC{self.star_data.TIC}_modelling_summary.csv",
                          index=False, header=True)
        self.amigo_res.amigo_res_df = summary_df

    def _on_click(self, event):
        '''
        Routine defining the actions performed on click.

        Parameters
        ----------
        event : mpl.event
            The mouse event happening in the plot.

        Returns
        -------
        None.

        '''
        # left click on pg axis: select periods.
        if event.inaxes == self.ax_pg and event.button == 1:
            self.picklistx.append(event.xdata)
            self.picklisty.append(event.ydata)
            self._update_plot()
        # right click on pg axis: add gap
        elif event.inaxes == self.ax_pg and event.button == 3:
            self.gaplist.append(event.xdata)
            self._update_plot()

    def _on_right_click(self, event):
        '''
        Defnines the events happening on right click.

        Parameters
        ----------
        event : mpl.event
            The right click event.

        Returns
        -------
        None.

        '''
        # if right right click in main panel
        if (event.button == 3 and event.inaxes == self.ax_pg
                and event.inaxes.get_navigate_mode() is None):

            print(self.ax_pg.get_navigate_mode())
            print(f"Right click at coordinates: ({event.xdata}, {event.ydata})")

            # if there are already gaps plotted find closest gap
            if len(self.gaplist) > 0:
                gap_arr = np.asarray(self.gaplist)
                idx = (np.abs(gap_arr - event.xdata)).argmin()
                closest_gap = gap_arr[idx]

                print(np.abs(closest_gap - event.xdata) / np.max([closest_gap, event.xdata]))
                # if closer than 1% remove gap
                if np.abs(closest_gap - event.xdata) / np.max([closest_gap, event.xdata]) < 0.01:
                    self.gaplist.remove(closest_gap)
                # else add a new gap
                else:
                    self.gaplist.append(event.xdata)

            else:
                self.gaplist.append(event.xdata)

            self._update_plot()

    def _on_pick(self, event):
        '''
        Defines the events happening on left click.

        Parameters
        ----------
        event : mpl.event
            The left click event.

        Returns
        -------
        None.

        '''
        print('clicked')

        ind = event.ind[0]

        print('onpick scatter:', ind, 1/self.good_freq[ind], self.good_amp[ind])
        ind = int(ind)
        pickedx = float(1 / self.good_freq[ind])
        pickedy = float(self.good_amp[ind])
        if ind in self.indpicklist:
            self.indpicklist.remove(ind)
            self.picklistx.remove(pickedx)
            self.picklisty.remove(pickedy)
        else:
            self.indpicklist.append(ind)
            self.picklistx.append(pickedx)
            self.picklisty.append(pickedy)

        self._update_plot()

    def _submit(self, _):
        '''
        Run AMiGO by clicking on the button.

        Returns
        -------
        None.

        '''
        command = self.text_box.text
        sortinds = np.argsort(self.picklistx)
        sorted_picklistx = np.array(self.picklistx)[sortinds]
        self._clear_ax(self.ax_psp)

        try:
            self.ax_psp.plot(sorted_picklistx[:-1], np.diff(sorted_picklistx) * 24 * 3600)
        except Exception as e:
            print(e)

        # fitting is happening here
        self._run_and_save_amigo(command)
        self.fitted = True

        # update plots
        self._update_plot()
        self._plot_chi2_space_live()

        self.fig.canvas.draw()

        # save current view for future reference
        self.fig.savefig(f"{self.pattern_path}TIC{self.star_data.TIC}_spacing.png")

    def _button_click_pass(self, _):
        '''
        Defines the events happning on clicking the pass button.

        Returns
        -------
        None.

        '''
        print("Pass button clicked!")
        self._flash_button(self.button_pass)

        # write to pass list and close
        with open(self.pattern_path + self.pass_list_name, 'a+', encoding="utf-8") as f:
            f.write(f'{self.star_data.TIC} \n')
        plt.close(self.fig)

    def _button_click_reset(self, _):
        '''
        Defines the events happning on clicking the reset button.

        Returns
        -------
        None.

        '''
        print("Reset button clicked!")
        self._flash_button(self.button_reset)

        # reset picked periods to empty lists
        self.picklistx = []
        self.picklisty = []
        self.indpicklist = []
        self.gaplist = []
        self.fitted = False

        # replot
        self._update_plot()

    def _button_click_softfail(self, _):
        '''
        Defines the events happning on clicking the softfail button.

        Returns
        -------
        None.

        '''
        print("Soft fail button clicked!")
        self._flash_button(self.button_softfail)

        # write to softfail list and close
        with open(self.pattern_path + self.softfail_list_name, 'a+', encoding="utf-8") as f:
            f.write(f'{self.star_data.TIC} \n')
        plt.close(self.fig)

    def _button_click_fail(self, _):
        '''
        Defines the events happning on clicking the fail button.

        Returns
        -------
        None.

        '''
        print("Fail button clicked!")
        self._flash_button(self.button_fail)

        # write to fail list and close
        with open(self.pattern_path + self.fail_list_name, 'a+', encoding="utf-8") as f:
            f.write(f'{self.star_data.TIC} \n')
        plt.close(self.fig)

    def _flash_button(self, button):
        '''
        UI effect when slecting a button.

        Parameters
        ----------
        button : mpl.button
            The button the mouse is on.

        Returns
        -------
        None.

        '''
        button.ax.set_facecolor('white')
        plt.pause(0.01)
        button.ax.set_facecolor(self.button_fail.hovercolor)
        button.ax.figure.canvas.draw()

    def _create_button(self, position, label, color, hovercolor, callback):
        '''
        Create a button with the properties given.

        Parameters
        ----------
        position : list-like
            List of length four with [xpos, ypos, xlen, ylen].
        label : str
            Text for the button.
        color : str
            A colour understood by matplotlib for the button
        hovercolor : str
            A colour understood by matplotlib for the button when activated.
        callback : function
            Function to be executed on click.

        Returns
        -------
        button : TYPE
            DESCRIPTION.

        '''
        ax = self.fig.add_axes(position)
        button = Button(ax, label)
        button.on_clicked(callback)
        button.color = color
        button.hovercolor = hovercolor
        return button

    def make_analysis(self, star_data, initial_amigo_kwarg='default'):
        '''
        Main public funtion of the class. This needs to be called to run the GUI.

        Parameters
        ----------
        star_data : StarData
            Object holding the observational data. See above.
        initial_amigo_kwarg : str, optional
            String of a dictionary holding the initial AMiGO parameters. This string will be shown
            in the box and later evaluated. Also except "default". The default is 'default'.

        Returns
        -------
        None.

        '''
        # set box content

        # add star_data to self
        self.star_data = star_data

        # get prev amigo results if available
        self.amigo_res = AMiGO_data(self.star_data.TIC, res_path=self.pattern_path)

        if (self.amigo_res.amigo_kwarg is not None) and (initial_amigo_kwarg == 'default'):
            initial_amigo_kwarg = self.amigo_res.amigo_kwarg
        elif initial_amigo_kwarg == 'default':
            initial_amigo_kwarg = ('{"frot_bnd": [1.5, 2, 0.005], "Pi0_bnd": [4500, 5600, 20],' +
                                   '"kval": 0, "mval": 1, "nthreads": 16}')

        # read all the available data and prepare the periodograms
        self._prepare_data()

        # exit if EB analysis return no non-harmonics
        if len(self.good_freq) == 0:
            print(f'TIC {self.star_data.TIC}: no non-harmonic frequencies. Skipping.')
            plt.close(self.fig)
            return

        # plot the periodogram and picked periods
        if self.amigo_res.obsstar is not None:
            self.fitted = True

        self._update_plot()

        # set axis limits
        self.ax_pg.set_ylim(0, max(self.good_amp) * 1.1)
        self.ax_pg.set_xlim(0, min([max(1 / self.good_freq), 2]))

        # plot LC before fitting
        self._plot_lc()

        # text box with amigo kwargs
        axbox = self.fig.add_axes([0.03, 0.43, 0.85, 0.04])
        self.text_box = TextBox(axbox, "", textalignment="center", initial=initial_amigo_kwarg)
        # text_box.on_submit(self._submit)

        # make buttons
        self.button_pass = self._create_button([0.88, 0.89, 0.1, 0.1], 'Pass', 'green',
                                               'lightgreen', self._button_click_pass)

        self.button_fail = self._create_button([0.02, 0.89, 0.1, 0.1], 'Fail', 'red', 'coral',
                                               self._button_click_fail)

        self.button_softfail = self._create_button([0.2, 0.89, 0.1, 0.1], 'Soft fail', 'orange',
                                                   'moccasin', self._button_click_softfail)

        self.button_reset = self._create_button([0.7, 0.89, 0.1, 0.1], 'Reset', 'cornflowerblue',
                                                'powderblue', self._button_click_reset)

        self.button_run_amigo = self._create_button([0.89, 0.43, 0.1, 0.04], 'Run AMiGO',
                                                    'mediumseagreen',
                                                    'lightgreen', self._submit)

        plt.show(block=True)

    def predict_pattern(self, frot_bnd=[0, 2.5, 0.005], Pi0_bnd=[2300, 5600, 100],
                        kval=0, mval=1, nthreads=1,
                        gyre_dir=None, method=None, diagnostic=None, use_sequence=None,
                        sig_sampling=None, grid_scaling=None, cvg_rate=None):
        '''
        Since AMiGO does not has a proper Python interface, we have to make it our own. This
        function takes the different parameters and passes them to the appropriate AMiGO functions
        to fit the period spacing pattern.

        For more information on all these properties see the AMiGO documentation.

        Parameters
        ----------
        frot_bnd : list, optional
            Fitting boundaries and step size for near-core rotation. The default is [0, 2.5, 0.005].
        Pi0_bnd : list, optional
            Fitting boundaries and step size for Pi0. The default is [2300, 5600, 100].
        all_kvals : list, optional
            k values to try to fit. The default is [1, 0].
        all_mvals : list, optional
            m values to try to fit. The default is [0, 1].
        nthreads : int, optional
            Number of threads for multiprocessing. The default is 1.
        gyre_dir : str, optional
            Path to the GYRE directory, otherwise read from the AMiGO defaults. The default is None.
        method : str, optional
            Set the fitting method "grid", or "iterative". See AMigo documentation.
            The default is None.
        diagnostic : str, optional
            Frequency or pattern. The default is None.
        use_sequence : str, optional
            See AMiGO documentation. The default is None.
        sig_sampling : str, optional
            See AMiGO documentation. The default is None.
        grid_scaling : str, optional
            See AMiGO documentation. The default is None.
        cvg_rate : float, optional
            See AMiGO documentation. The default is None.

        Returns
        -------
        None.

        '''
        # star=self.star_data.name, obsfile=self.amigo_res.pattern_path,
        # output_dir=self.pattern_path

        # avoiding unnecessary warnings
        warnings.simplefilter('ignore')

        # read in the chosen parameters and relevant filenames in the inlist
        config_file = files(amigo).joinpath('../defaults/config.dat')
        def_num = files(amigo).joinpath('../defaults/default_numerical.dat')

        # Reading in the main config files
        with open(config_file, 'r', encoding="utf-8") as file:
            lines = file.readlines()

        # A dictionary of all the possible sections in the inlist, except 'modes'
        inlist_params = {'observations': {}, 'numerical': {}, 'star': {},
                         'rotation': {}, 'output': {}}

        # Reading in the default AMiGO numerical settings
        inlist_params = compute_rotation.read_routine(def_num, inlist_params)

        # assign default values from config files if no values provided.
        if gyre_dir is None:
            for line in lines:
                if line.split('=')[0].strip() == 'gyre_dir':
                    gyre_dir = line.split('=')[1].split('#')[0].strip()

        if method is None:
            method = inlist_params['numerical']['optimisation_method']
        if diagnostic is None:
            diagnostic = inlist_params['numerical']['diagnostic']
        if use_sequence is None:
            use_sequence = bool(inlist_params['numerical']['use_sequence'])
        if sig_sampling is None:
            sig_sampling = float(inlist_params['numerical']['sigma_sampling'])
        if grid_scaling is None:
            grid_scaling = float(inlist_params['numerical']['grid_scaling'])
        if cvg_rate is None:
            cvg_rate = float(inlist_params['numerical']['cvg_rate'])

        # reading in the observations
        self.amigo_res.obsstar = obspuls(self.star_data.name, self.amigo_res.pattern_path)

        # preparing the output directory and log file if the path in the inlist exists
        if os.path.exists(self.pattern_path):
            if not os.path.exists(f'{self.pattern_path}/{self.star_data.name}'):
                os.makedirs(f'{self.pattern_path}/{self.star_data.name}')
            output_log = True
        else:
            output_log = False

        # Fitting the psp with the given mode ID

        log_str = (f"fitting (k,m) = ({kval}, {mval}) to the observed g-modes of" +
                   f" {self.star_data.name}:")
        print(log_str)

        # Preparing the asymptotic g-mode models
        # In contrast to the full AMiGO fitting routine, we only fit one set of (k, m) as most
        # of the stars are pro-grade dipole modes.
        self.amigo_res.modes = [gravity_modes(gyre_dir, kval=kval, mval=mval)]

        # Calculating the asymptotic fits and plotting the results
        if method == 'grid':
            amigo_res = fit.fit_with_grid(self.amigo_res.obsstar, self.amigo_res.modes,
                                          diagnostic=diagnostic,
                                          Pi0min=Pi0_bnd[0], Pi0max=Pi0_bnd[1], dPi0=Pi0_bnd[2],
                                          frotmin=frot_bnd[0], frotmax=frot_bnd[1],
                                          dfrot=frot_bnd[2], alpha_in=None,
                                          use_sequence=use_sequence, nthreads=nthreads)

        elif method == 'iterative':
            amigo_res = fit.fit_iterative(self.amigo_res.obsstar, self.amigo_res.modes,
                                          diagnostic=diagnostic,
                                          Pi0min=Pi0_bnd[0], Pi0max=Pi0_bnd[1], dPi0=Pi0_bnd[2],
                                          frotmin=frot_bnd[0], frotmax=frot_bnd[1],
                                          dfrot=frot_bnd[2], alpha_in=None,
                                          use_sequence=use_sequence,
                                          nthreads=nthreads, sigma_sampling=sig_sampling,
                                          grid_scaling=grid_scaling, cvg_rate=cvg_rate,
                                          output_log=output_log)

        else:
            amigo_res = fit.fit_with_lmfit(self.amigo_res.obsstar, self.amigo_res.modes,
                                           diagnostic=diagnostic,
                                           Pi0min=Pi0_bnd[0], Pi0max=Pi0_bnd[1],
                                           frotmin=frot_bnd[0], frotmax=frot_bnd[1],
                                           alpha_in=None, use_sequence=use_sequence,
                                           nthreads=nthreads)

        # Printing the final results
        fin_frot, fin_e_frot, fin_Pi0, fin_e_Pi0, alpha_out, e_alpha, \
            min_chi2red, _, _, _, _ = amigo_res

        print(f"    chi2_red = {min_chi2red}")
        print(f"    f_rot: {fin_frot.value} +/- {fin_e_frot.value} c/d")
        print(f"    Pi0: {fin_Pi0.value} +/- {fin_e_Pi0.value} s")

        if output_log:
            logpath = (f'{self.pattern_path}/{self.star_data.name}/' +
                       f'{self.star_data.name}_modelling.log')
            with open(logpath, 'w+', encoding="utf-8") as logfile:
                logfile.write(f"\n{log_str}\n")
                logfile.write(f"chi2_red = {min_chi2red}\n")
                logfile.write(f"f_rot: {fin_frot.value} +/- {fin_e_frot.value} c/d\n")
                logfile.write(f"Pi0: {fin_Pi0.value} +/- {fin_e_Pi0.value} s\n")

                if diagnostic == 'frequency':
                    for ii, ialpha, e_ialpha in zip(np.arange(1, 1+len(alpha_out)),
                                                    alpha_out, e_alpha):
                        logfile.write(f"    alpha{int(ii)}: {ialpha} +/- {e_ialpha}\n")
                else:
                    for ii, ialpha in zip(np.arange(1, 1+len(alpha_out)), alpha_out):
                        logfile.write(f"    estimated alpha{int(ii)}: {ialpha}\n")
                logfile.write("\n")

        # assigning values to self.amigo_res
        self.amigo_res.frot = fin_frot.value
        self.amigo_res.frot_e = fin_e_frot.value
        self.amigo_res.Pi0 = fin_Pi0.value
        self.amigo_res.Pi0_e = fin_e_Pi0.value
        self.amigo_res.alpha = alpha_out[0]
        self.amigo_res.chi_sq = min_chi2red
        self.amigo_res.sample = amigo_res[8]
        self.amigo_res.chi2_full = amigo_res[9]
        self.amigo_res.deg_freedom = amigo_res[10]

        self.amigo_res.gyre_dir = gyre_dir
