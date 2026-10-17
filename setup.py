"""TEACLIPPER

Code written by: ...
"""

from setuptools import setup


# package version
MAJOR = 1
MINOR = 1
ATTR = 8
# full acronym
ACRONYM = ('TEss Asteroseismic Combined LIght curve Pipeline for PERiod spacings')

setup(name="teaclipper",
      version=f'{MAJOR}.{MINOR}.{ATTR}',
      author='...',
      url='https://github.com/freestket/teaclipper',
      # license='GNU General Public License v3.0',
      description=ACRONYM,
      long_description=open('README.md').read(),
      long_description_content_type='text/markdown',
      packages=['teaclipper'],
      package_dir={'teaclipper': 'teaclipper'},
      python_requires='>=3.6,<3.13',
      install_requires=[
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