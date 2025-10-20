#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  13 13:35:16 2025

@author: dario with the help of AI (dario.fritzewski@kuleuven.be)

This is a tool to plot light curves and periodograms to facilitate the
visual classification.

Usage: LCcheckGUI.launch_lc_review_gui(TICs, LCdir=<path to your TU reduction>,
                                       red_name=<add name string>)

It will show a window with all the LCs in the top panel. This panel is scollable.
The top matplotlib toolbar works for the light curves.
In the bottom two periodograms are shown in frequency and period.
At the very bottom, you find a drop-down menu for classification. You can add your
own string if needed. The comment field allows you to add comments for later.
Hit the button or <Enter> to continue to the next LC. Previously classified LCs
should be skipped automatically. If you want to reclassify  give a different
"reviewfile" name.

"""
import os
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from . import utils


def launch_lc_review_gui(TICs, reviewfile='review_results.csv', LCdir='./TU/',
                         version='detrended', red_name=''):
    """
    Main function to launch the GUI

    Parameters
    ----------
    TICs : list-like
        List of TIC IDs to inspect LCs of.
    reviewfile : str, optional
        Path to the csv file that saves the comments.
        The default is 'review_results.csv'.
    LCdir : str, optional
        Path to the light curve directory. The default is './TU/'.
    version : str, optional
        Version of the reduction, can either be 'TU' or 'detrended'. Use the latter
        if you used the detrending in tea_clipper. The default is 'detrended'.
    red_name : str, optional
        Add string attched to the reduction of TU. The default is ''.

    Returns
    -------
    None.

    """

    reviewed_tics = set()
    if os.path.exists(reviewfile):
        reviewed_tics = set(pd.read_csv(reviewfile)['TIC'])

    TICs = [TIC for TIC in TICs if TIC not in reviewed_tics]

    class ReviewApp:
        """
        The main app that allows you to review the LCs.
        """

        def __init__(self, master, TICs, reviewfile='review_results.csv',
                     LCdir='./TU/', version='detrended', red_name=''):
            self.master = master
            self.TICs = TICs
            self.outname = reviewfile
            self.index = 0

            self.LCdir = LCdir
            self.version = version
            self.red_name = red_name

            # New instance variable to store the canvas item ID
            self.scrollable_frame_id = None

            self.master.title("Light Curve Inspection Tool")

            # set window size
            self.master.geometry("1200x1350")

            # Main layout frame
            self.main_frame = tk.Frame(master)
            self.main_frame.pack(fill=tk.BOTH, expand=True)

            # configure grid weights
            # Make the single column expand horizontally
            self.main_frame.grid_columnconfigure(0, weight=1)

            # place toolbar above scrolable area
            self.lc_toolbar_frame = tk.Frame(self.main_frame)
            self.lc_toolbar_frame.grid(row=0, column=0, sticky="ew")

            self.main_frame.grid_rowconfigure(0, weight=0)
            # Row 0: LCs
            self.main_frame.grid_rowconfigure(1, weight=8)
            # Row 1: Periodograms
            self.main_frame.grid_rowconfigure(2, weight=2)
            # Row 2: Controls
            self.main_frame.grid_rowconfigure(3, weight=0)

            # --- 1. Scrollable frame for Light Curves (LCs) ---
            self.lc_canvas_frame = tk.Frame(self.main_frame)
            self.lc_canvas_frame.grid(row=1, column=0, sticky="nsew")

            # Frame to hold the actual canvas and scrollbar
            self.canvas = tk.Canvas(self.lc_canvas_frame)
            self.scrollbar = tk.Scrollbar(self.lc_canvas_frame, orient="vertical",
                                          command=self.canvas.yview)

            # Frame to hold LC plots - this is what scrolls
            self.scrollable_frame = tk.Frame(self.canvas)

            # Bind events for scrolling and width
            self.scrollable_frame.bind(
                "<Configure>",
                lambda e: self.canvas.configure(
                    scrollregion=self.canvas.bbox("all")
                )
            )
            # Bind the canvas to the main window's width for dynamic resizing
            self.canvas.bind('<Configure>', self._on_canvas_resize)

            # Store the item ID when creating the window
            self.scrollable_frame_id = self.canvas.create_window(
                (0, 0), window=self.scrollable_frame, anchor="nw"
            )

            self.canvas.configure(yscrollcommand=self.scrollbar.set)

            self.canvas.pack(side="left", fill="both", expand=True)
            self.scrollbar.pack(side="right", fill="y")

            # --- 2. Non-scrollable frame for Periodograms (PGs) ---
            self.pg_frame = tk.Frame(self.main_frame)
            # Place in Row 1
            self.pg_frame.grid(row=2, column=0, sticky="ew")

            # --- 3. Controls frame (Classification, Comment, Save) ---
            self.controls_frame = tk.Frame(self.main_frame)
            # Place in Row 2
            self.controls_frame.grid(row=3, column=0, sticky="ew")

            self._setup_controls()
            self.load_data()

        def _on_canvas_resize(self, event):
            """
            Dynamic Width for LCs: Adjust the width of the frame inside the canvas
            when the canvas size changes. Use the stored ID.

            Parameters
            ----------
            event : tk event
                Resizing event.

            Returns
            -------
            None.

            """
            # Dynamic Width for LCs: Adjust the width of the frame inside the canvas
            # when the canvas size changes. Use the stored ID.
            canvas_width = event.width
            if self.scrollable_frame_id:
                self.canvas.itemconfig(self.scrollable_frame_id, width=canvas_width)

        def _setup_controls(self):
            """
            Set up the controls in the window.

            Returns
            -------
            None.

            """
            # Setup widgets in self.controls_frame
            self.class_label = ttk.Label(self.controls_frame, text="Classification:")
            self.class_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
            self.class_var = tk.StringVar()
            self.class_dropdown = ttk.Combobox(self.controls_frame, textvariable=self.class_var,
                                               width=15)
            self.class_dropdown['values'] = ['asteroseismic signal', 'no signal', 'redo photometry']
            self.class_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky='w')

            self.comment_label = ttk.Label(self.controls_frame, text="Comment:")
            self.comment_label.grid(row=0, column=2, padx=5, pady=5, sticky='w')

            # --- Comment Entry Widget ---
            self.comment_entry = ttk.Entry(self.controls_frame, width=60)
            self.comment_entry.grid(row=0, column=3, padx=5, pady=5, sticky='w')

            # When Enter is hit while the text box is focused, call save_and_next
            self.comment_entry.bind('<Return>', lambda event: self.save_and_next())

            self.save_button = ttk.Button(self.controls_frame, text="Save & Next",
                                          command=self.save_and_next)
            self.save_button.grid(row=0, column=4, padx=5, pady=5, sticky='w')

            # Make columns expandable
            self.controls_frame.grid_columnconfigure(3, weight=1)

        def load_data(self):
            """
            Load data and create all the plots. Could be refactored but it works...

            Returns
            -------
            None.

            """
            # Clear previous plots/widgets
            for widget in self.scrollable_frame.winfo_children():
                widget.destroy()
            for widget in self.pg_frame.winfo_children():
                widget.destroy()
            # Clear the FIXED LC toolbar frame before adding a new toolbar
            for widget in self.lc_toolbar_frame.winfo_children():
                widget.destroy()

            TIC = self.TICs[self.index]
            current_number = self.index + 1
            try:
                lc = utils.load_lc(TIC, LCdir=self.LCdir, version=self.version,
                                   red_name=self.red_name, return_lk=True)
            except FileNotFoundError:
                print(f'TIC {TIC}: no lc found.')
                self.index += 1
                return

            pg = lc.to_periodogram()
            sector_borders, sectors = utils.get_sectors_dates(lc.time.value, return_numbers=True)

            s_time = np.split(lc.time, sector_borders)
            s_flux = np.split(lc.flux, sector_borders)

            nSec = len(sectors)

            # Light Curve Plots
            rows = (nSec + 1) // 2
            fig_lc, axs = plt.subplots(rows, 2, figsize=(10, 2.5 * rows))
            axs = axs.flatten()

            for i in range(nSec):
                axs[i].plot(s_time[i].value, s_flux[i].value, color='black', lw=0.5)
                axs[i].set_xlabel('Time [MBJD]')
                axs[i].set_ylabel('Norm. flux')
                axs[i].set_title(f'Sector {sectors[i]}')
                utils.make_ticks(axs[i])
            for j in range(nSec, len(axs)):
                axs[j].axis('off')

            fig_lc.tight_layout()
            # --- Light Curve Toolbar Placement ---
            # Place the canvas in the scrollable frame
            canvas_lc = FigureCanvasTkAgg(fig_lc, master=self.scrollable_frame)
            canvas_lc.draw()
            canvas_lc.get_tk_widget().pack(fill=tk.X, expand=True)

            # Create the toolbar and place it in the NEW, FIXED frame
            toolbar_lc = NavigationToolbar2Tk(canvas_lc, self.lc_toolbar_frame)
            toolbar_lc.update()
            # Use pack to place the toolbar within the toolbar frame
            toolbar_lc.pack(side=tk.TOP, fill=tk.X)

            plt.close(fig_lc)  # Close the matplotlib figure to free memory

            # --- Periodogram Plots (Non-Scrollable Area) ---
            fig_pg, ax_pg = plt.subplots(2, 1, figsize=(10, 4))
            ax_pg[0].plot(pg.frequency.value, pg.power.value * 1000, color='k', lw=0.1)
            ax_pg[0].set_title(f'TIC {TIC} ({current_number}/{len(self.TICs)})')
            ax_pg[0].set_xlabel('Frequency [d$^{-1}$]')
            ax_pg[0].set_ylabel('Amplitude [ppt]')
            ax_pg[0].set_xlim(0.5, 30)
            ax_pg[0].set_ylim(0, None)

            ax_pg[1].plot(pg.period.value, pg.power.value * 1000, color='k', lw=0.1)
            ax_pg[1].set_xlabel('Period [d]')
            ax_pg[1].set_ylabel('Amplitude [ppt]')
            ax_pg[1].set_xlim(0, 1.5)
            ax_pg[1].set_ylim(0, None)

            for a in ax_pg:
                utils.make_ticks(a)

            fig_pg.tight_layout()
            canvas_pg = FigureCanvasTkAgg(fig_pg, master=self.pg_frame)
            canvas_pg.draw()

            # Add Navigation Toolbar
            toolbar_pg = NavigationToolbar2Tk(canvas_pg, self.pg_frame)
            toolbar_pg.update()

            # Use fill=tk.X for horizontal expansion
            canvas_pg.get_tk_widget().pack(fill=tk.X, expand=True)
            plt.close(fig_pg)  # Close the matplotlib figure

            # Update the scroll region
            self.scrollable_frame.update_idletasks()
            self.canvas.config(scrollregion=self.canvas.bbox("all"))

        def save_and_next(self):
            """
            Save the comments and classification to the csv file.
            Returns
            -------
            None.

            """
            classification = self.class_var.get()
            comment = self.comment_entry.get()
            tic_id = self.TICs[self.index]

            if classification == '':
                messagebox.showwarning("Warning", "Please select a classification.")
                return
            if classification not in self.class_dropdown['values']:
                self.class_dropdown['values'] = (*self.class_dropdown['values'], classification)

            # Append to review_results.csv
            pd.DataFrame({'TIC': [tic_id],
                          'Classification': [classification],
                          'Comment': [comment]}).to_csv(self.outname,
                                                        mode='a',
                                                        header=not os.path.exists(self.outname),
                                                        index=False)

            self.index += 1
            self.comment_entry.delete(0, tk.END)
            self.class_var.set('')

            if self.index >= len(self.TICs):
                messagebox.showinfo("Done", "All TICs reviewed. Data saved.")
                self.master.quit()
            else:
                self.load_data()

    root = tk.Tk()
    _ = ReviewApp(root, TICs, reviewfile=reviewfile, LCdir=LCdir,
                  version=version, red_name=red_name)
    root.mainloop()
