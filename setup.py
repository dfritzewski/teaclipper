"""TEACLIPPER

Code written by: ... 
""" # update author

from setuptools import setup


# package version
MAJOR = 1
MINOR = 0
ATTR = 1

# full acronym
ACRONYM = ('TEss Asteroseismic Combined LIght curve Pipeline for PERiod spacings')

setup(name="teaclipper",
      version=f'{MAJOR}.{MINOR}.{ATTR}',
      author='...',                                   # update author
      url='https://github.com/freestket/teaclipper',  # update to correct package link
      # license='...',
      description=ACRONYM,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=['teaclipper'],
      package_dir={'teaclipper': 'teaclipper'},
      package_data={'teaclipper': ['TESS_orbit_times.csv']},# include the .csv file in the package install
      python_requires='>=3.11,<3.12',                       # update python dependency
      install_requires=[                                    # update or verify package dependencies / add versions
            'astropy',
            'astroquery',
            'joblib',
            'lightkurve',
            'matplotlib',
            'numpy',
            'pandas',
            'peakutils',
            'scipy',
            'tqdm',
            'IPython',
            'ipympl',
            'numba',
            'h5py',
            'arviz',
            'corner',
            'lmfit',
            'tk'
        ],
     )