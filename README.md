Tea clipper: TEss Asteroseismic Combined LIght curve Pipeline for PERiod spacings
================================================================

Author: Dario Fritzewski (dario.fritzewski@kuleuven.be)

Tea clipper is a Python package combining various Python packages developed at the asteroseismology group at the Institute of Astronomy (IvS) at KU Leuven to enable a smooth workflow for gravity mode asteroseismology. The aim of tea clipper is to provide a unified interface to the main three packages:
1. `tessutils` for obtaining light curves from TESS full-frame images (https://github.com/IvS-Asteroseismology/tessutils)
2. `star shadow` as the pre-whitening code (https://github.com/IvS-Asteroseismology/star_shadow)
3. `AMiGO` for fitting detected period-spacing patterns (https://github.com/IvS-Asteroseismology/amigo)

In addition, to the unified I/O to these packages, tea clipper offers two GUIs for light curve analysis and period spacing pattern identification and fitting.

Workflow
------------
To search for period-spacing patterns with TESS, we need stars with many observed sectors. Ideal stars are located in the TESS continuous viewing zones. We choose TIC ## form [Garcia et al (2022)](https://ui.adsabs.harvard.edu/abs/2022A%26A...662A..82G/abstract).

1. Download the TESS data and create light curve

```
from teaclipper import TUcluster
TICs = [##]
TUcluster.run_tu(TICs)
```


Now there should be a folder called `TU` in your current directory. It contains the light curve and other intermediate products. For a full documentation of `tessutils` capabilities please consult its [github page](https://github.com/IvS-Asteroseismology/tessutils).

2. To inspect the light curve beyond the diagnosis plot produced by `tessutils` (which you should consult to check for example the contamination), we can use the light curve GUI:

```
from teaclipper import LCcheckGUI
LCcheckGUI.launch_lc_review_gui(TICs, version='TU')
```

A tk python window should appear. At the top, it shows all the light curves per Sector. Scroll down to see all light curves. The tool bar at the top allows the user to interact with the light curve plots
In the bottom part, the periodogram is shown both in frequency and period space, to facilitate the asteroseismic analysis. Similarly to the light curves the tool bar at the bottom allows for interaction with these plots.

3. Even after the detrending with the principal component analysis in `tessutils`, some trends persist that we can remove with polynomial detrending. 

```
from teaclipper import detrendLCs
detrendLCs.all_poly_detrend(TICs)
```

4. Call the light curve GUI again to check if the light curves have improved. Now you can sort your light curves according to its asteroseismic potential.

```
LCcheckGUI.launch_lc_review_gui(TICs, red_name=red_name)
```

At the very bottom a drop down menu allows to select a category (or add a custom one) and add comment for each TIC (in the case TICs, holds more than one ID). The results are saved to a csv file for easy filtering. 

5. Now its time to extract all asteroseismicly relevant frequencies with `starshadow`:
```
from teaclipper import prewhitenSTS
prewhitenSTS.prewhiten_all(TICs, red_name=red_name)
```


This runs `starshadow` and creates all relevant files in the folder `STS`.

6. Finally, we can extract the gravity mode period spacing pattern and analyse it with `AMiGO`.

```
from teaclipper import findPatterns
findPatterns.analyse_all(TICs, red_name=red_name)
``` 

A different GUI should appear. The main plot shows the periodogram including the pre-whitened frequencies. By clicking on them they can be selected. Once more than one frequency is selected the bottom left plot shows the period spacing pattern. Right click allows to add gaps in the pattern. After selecting the detected pattern, the `AMiGO` parameters in the text box can be adjusted. By clicking "Run AMiGO" the fitting routine is initiated. Once a good fit is achieved (or not), the pass (fail) button at the top allow to go the next star in `TICs`.

This workflow is also outlined in `example.py`.

Installation instructions
------------------------------

Tea clipper needs a running installation of `tessutils`, `starshadow`, and `AMiGO`, in addition it required `tk`. The `requirements.txt` includes all packages required to run tea clipper and `tessutils`, `starshadow`, and `AMiGO`. For problems with installing any of the other IvS packages please consult their respective github pages.