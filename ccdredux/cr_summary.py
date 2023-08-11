"""
Code to print a summary of the CRVAL and CRPIX values in a file or set of files

Usage: python cr_summary.py [fitsfile(s)]

Input parameters:
    fitsfile(s) - Either a single filename or a wildcard expression
                   e.g.,  m13*fits
"""

import sys
import numpy as np

from astropy import wcs
from astropy.io import fits as pf
from matplotlib import pyplot as plt

from specim import imfuncs as imf
from ccdredux.ccdset import CCDSet

""" Check command line syntax """
if len(sys.argv)<2:
    print('')
    print('Usage:')
    print(' python cr_summary.py [fitsfile(s)]')
    print('')
    print('Required inputs:')
    print(' fitsfile(s) - Either a single filename or a wildcard expression')
    print('  e.g.,  m13*fits')
    print('')
    exit()

""" Create the input file list """
if len(sys.argv) > 2:
   files = sys.argv[2:]
else:
    files = [sys.argv[2],]

""" Load the data into a CCDSet object """
imdat = CCDSet(files)

""" Print out the summary of the CRPIX and CRVAL values """
imdat.print_cr_summary()

""" Clean up """
del imdat
