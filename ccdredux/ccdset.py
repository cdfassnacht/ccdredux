"""

Defines a CCDSet class that can be used for the standard calibration steps
for CCD and similar data sets
"""

import sys
from os import path
import numpy as np
from math import floor, sqrt

from astropy import wcs
from astropy.io import fits as pf
from astropy.io import registry
from astropy.table import Table
from scipy.ndimage import filters
from matplotlib import pyplot as plt

from specim.imfuncs import WcsHDU, Image, imfit
from specim.imfuncs.dispparam import DispParam
from specim.imfuncs.dispim import DispIm

pyversion = sys.version_info.major


# ===========================================================================

class CCDSet(list):
    """

    A class that can be used to perform standard calibration steps for
    a collection of CCD and CCD-like data sets

    """

    def __init__(self, inlist, hext=0, wcsext=None, indir=None,
                 filecol='infile', tabformat=None, infokeys=None, texpkey=None,
                 gainkey=None, wcsverb=False, prefix=None, suffix=None,
                 verbose=True, **kwargs):
        """

        Creates a CCDSet object by reading in the relevant data sets

        """

        """ Set up the empty CCDSet container by calling the superclass """
        if pyversion == 2:
            super(list, self).__init__()
        else:
            super().__init__()

        """ Set some informational parameters """
        self.nfiles = len(inlist)
        self.hext = hext
        if wcsext is not None:
            self.wcsext = wcsext
        else:
            self.wcsext = hext

        """ Set default values """
        self.datainfo = None
        self.bias = None
        self.bpmglobal = None
        self.flat = None
        self.fringe = None
        self.darkskyflat = None
        self.objmasks = None
        
        """ Set up for loading the data """
        if isinstance(inlist, (list, tuple)):
            self.datainfo = Table()
            filelist = []
            if isinstance(inlist[0], str):
                for f in inlist:
                    if prefix is not None:
                        name = '%s%s' % (prefix, str(f))
                    else:
                        name = str(f)
                    if suffix is not None:
                        name = '%s%s' % (name, suffix)
                    if indir is not None:
                        name = path.join(indir, name)
                    filelist.append(name)
                self.datainfo['infile'] = filelist
            elif isinstance(inlist[0],
                            (pf.PrimaryHDU, pf.ImageHDU, WcsHDU, Image)):
                for hdu in inlist:
                    if hdu.infile is not None:
                        filelist.append(hdu.infile)
                    else:
                        filelist.append('N/A')
                self.datainfo['infile'] = filelist
        elif isinstance(inlist, Table):
            if filecol is None:
                raise ValueError('Input list is a Table but filename column '
                                 'has not been given\n'
                                 'Please set the "filecol" parameter')
            elif filecol not in inlist.keys():
                raise KeyError('Column %s not found in input table' % filecol)
            elif filecol != 'infile':
                inlist.rename_column(filecol, 'infile')
            self.datainfo = inlist.copy()
        elif isinstance(inlist, str):
            if tabformat is None or tabformat == 'fits':
                try:
                    self.datainfo = Table.read(inlist)
                except (IOError, registry.IORegistryError):
                    print('')
                    print('Could not load data table.  Check input format!')
                    print('')
            else:
                if tabformat[:5] != 'ascii':
                    tabformat = 'ascii.%s' % tabformat
                try:
                    self.datainfo = \
                        Table.read(inlist, guess=False, format=tabformat)
                except (IOError, registry.IORegistryError):
                    print('')
                    print('Could not load data table.  Check input format!')
                    print('')
            if filecol not in self.datainfo.keys():
                raise KeyError('Column %s not found in input table' % filecol)
            elif filecol != 'infile':
                self.datainfo.rename_column(filecol, 'infile')
        else:
            raise TypeError('\nERROR: Input is not in a valid format\n')
        self.datainfo['basename'] = self.datainfo['infile'].copy()

        """ Load the data into the object """
        if verbose:
            # print('')
            print('Loading data...')
        for f, info in zip(inlist, self.datainfo):
            if isinstance(f, (pf.PrimaryHDU, pf.ImageHDU, WcsHDU, Image)):
                infile = f
            else:
                infile = info['infile']
            tmp = WcsHDU(infile, hext=hext, wcsext=wcsext, verbose=False,
                         wcsverb=wcsverb, **kwargs)
            if tmp.basename is not None:
                info['basename'] = tmp.basename
            self.append(tmp)

        """ Put the requested information into datainfo table """
        keylist = ['object']
        if texpkey is not None:
            keylist.append(texpkey)
        else:
            texpkey = 'exptime'
            keylist.append('exptime')
        if gainkey is not None:
            keylist.append(gainkey)
        else:
            gainkey = 'gain'
            keylist.append('gain')
        if infokeys is not None:
            for key in infokeys:
                kl = key.lower()
                if kl != 'object' and kl != texpkey and kl != gainkey:
                    keylist.append(key)
        self.read_infokeys(keylist, texpkey=texpkey, gainkey=gainkey)

        """ Rename special columns if they are there """
        if texpkey in keylist:
            if texpkey.lower() != 'texp':
                self.datainfo.rename_column(texpkey, 'texp')
                keylist.append('texp')
        else:
            self.datainfo['texp'] = -1.
        if gainkey in keylist:
            if gainkey.lower() != 'gain':
                self.datainfo.rename_column(gainkey, 'gain')
                keylist.append('gain')
        else:
            self.datainfo['gain'] = -1.
        for k in ['texp', 'gain']:
            if isinstance(self.datainfo[k][0], float):
                self.datainfo[k].format = '%.2f'

        """ Summarize the inputs """
        if verbose:
            print('')
            self.print_summary(keylist)
            
    # -----------------------------------------------------------------------

    def read_infokeys(self, infokeys, texpkey, gainkey):
        """

        Adds information that is designated by the passed keywords to the
        datainfo table

        """

        """ Start by adding appropriate columns to the table """
        for key in infokeys:
            if key == texpkey or key == gainkey:
                self.datainfo[key] = -1
            else:
                self.datainfo[key] = None

        """ Get the information from the fits headers, if available """
        for hdu, info in zip(self, self.datainfo):
            hdr = hdu.header
            for key in infokeys:
                if key.upper() in hdr.keys():
                    info[key] = hdr[key.upper()]
                elif key == 'exptime' or key == 'gain':
                    infokeys.remove(key)
                elif key == texpkey or key == gainkey:
                    info[key] = -1
                else:
                    info[key] = 'N/A'

        """ Set the format for printing """
        for key in infokeys:
            if isinstance(self.datainfo[key][0], float):
                self.datainfo[key].format = '%.2f'

    # -----------------------------------------------------------------------

    def print_summary(self, infocols):
        """

        Summarizes the input file characteristics

        """

        sumkeys = ['basename']
        for k in infocols:
            if k in self.datainfo.keys():
                sumkeys.append(k)
        if len(sumkeys) > 0:
            infotab = self.datainfo[sumkeys]
            print(infotab)

    # -----------------------------------------------------------------------

    def print_cr_summary(self):
        """

        Prints a summary of the CRPIX and CRVAL values

        """

        print('')
        print('File                       CRVAL1      CRVAL2     CRPIX1 '
              '  CRPIX2     Notes')
        print('------------------------ ----------- ----------- --------'
              ' -------- -------------')
        for i, hdu in enumerate(self):
            hdr = hdu.header
            if hdu.basename is not None:
                f = hdu.basename[:-5]
            else:
                f = 'Image %d' % i
            print('%-24s %11.7f %+11.7f %8.2f %8.2f' %
                  (f, hdr['crval1'], hdr['crval2'], hdr['crpix1'],
                   hdr['crpix2']))
            del hdr

    # -----------------------------------------------------------------------

    def rms_summary(self, statradec, statsize, centtype='radec', sizetype=None):
        """
        Calculates the rms in a specifiec region (typically defined to be
        centered at the same (RA,Dec)) for the files in this object
        """

        print('')
        print('File             xcent   ycent     rms')
        print('--------------- ------- -------  --------')

        var = np.zeros(len(self))
        for i, hdu in enumerate(self):
            tmp = hdu.get_rms(statradec, statsize, centtype, sizetype,
                              verbose=False)
            if hdu.basename is not None:
                f = hdu.basename[:-5]
            else:
                f = 'Image %d' % i
            print('%-15s %7.2f %7.2f  %7f' %
                  (f, tmp['statcent'][0], tmp['statcent'][1], tmp['rms']))
            var[i] = tmp['rms']**2
            del tmp

        print('')
        rms_est = sqrt(var.sum()) / len(self)
        print('Estimated rms if files are averaged: %7f' % rms_est)
        print('')

    # -----------------------------------------------------------------------

    def read_calfile(self, filename, file_description, caldir=None, hext=0,
                     verbose=True):
        """

        Reads in a calibration file

        """
        if caldir is not None:
            infile = path.join(caldir, filename)
        else:
            infile = filename
        if verbose:
            print('Reading %s file: %s' % (file_description, infile))
            
        try:
            calhdu = WcsHDU(infile, hext=hext, verbose=False, wcsverb=False)
        except FileNotFoundError:
            print(' ERROR: Requested %s file %s does not exist' % 
                  (file_description, infile))
            print('')
            return None
        except OSError:
            print(' ERROR reading file %s' % infile)
            return None
        return calhdu

    # -----------------------------------------------------------------------

    def load_calib(self, biasfile=None, flatfile=None, fringefile=None,
                   darkskyfile=None, bpmglobfile=None, caldir=None, hext=None,
                   verbose=True, headverbose=True):
        """

        Loads external calibration files and stores them as attributes of
        the class

        """

        """
        Use the default HDU extension unless an override was requested
        """
        if hext is None:
            hext = self.hext
            
        """ Read in calibration frames if they have been selected """
        if headverbose:
            print('Loading any requested calibration files')
        if biasfile is not None:
            self.bias = self.read_calfile(biasfile, 'bias/dark', hext=hext,
                                          caldir=caldir, verbose=verbose)
            if self.bias is None:
                raise OSError('Error reading %s' % biasfile)

        if bpmglobfile is not None:
            self.bpmglobal = \
                self.read_calfile(bpmglobfile, 'global bpm',  hext=hext,
                                  caldir=caldir, verbose=verbose)
            if self.bpmglobal is None:
                raise OSError('Error reading %s' % bpmglobfile)

        if flatfile is not None:
            self.flat = self.read_calfile(flatfile, 'flat-field', hext=hext,
                                          caldir=caldir)
            if self.flat is None:
                raise OSError('Error reading %s' % flatfile)

        if fringefile is not None:
            self.fringe = self.read_calfile(fringefile, 'fringe', caldir=caldir,
                                            hext=hext)
            if self.fringe is None:
                raise OSError('Error reading %s' % fringefile)

        if darkskyfile is not None:
            self.darkskyflat = self.read_calfile(darkskyfile, 'dark-sky flat',
                                                 caldir=caldir, hext=hext)
            if self.darkskyflat is None:
                raise OSError('Error reading %s' % darkskyfile)

    # -----------------------------------------------------------------------

    def median_combine(self, outfile=None, outobj=None, method='median',
                       framemask=None,
                       trimsec=None, biasfile=None, flatfile=None,
                       usegain=False, usetexp=False, normalize=None,
                       zerosky=None, use_objmask=False, NaNmask=False,
                       verbose=True, headverbose=True):
        """ 
        This is one of the primary methods of the CCDSet class.  It will:

          1. Subtract a bias frame (if the optional biasfile parameter is set)
          2. Multiply by the gain, required to be in e-/ADU (if the optional
              gain parameter is set)
          3. Normalize the frame (if the optional normalize parameter is set)
          4. Subtract the median (if the optional zeromedian parameter is set)
          5. Median combine the resulting data
          6. Write the output to a file if requested

        """

        """ Load any requested calibration files """
        self.load_calib(biasfile, flatfile, verbose=verbose,
                        headverbose=headverbose)

        """
        Set up the container to hold the data stack that will be used to
        compute the median
        """
        if trimsec is not None:
            xsize = trimsec[2] - trimsec[0]
            ysize = trimsec[3] - trimsec[1]
        else:
            xsize = self[0].data.shape[1]
            ysize = self[0].data.shape[0]
        if framemask is not None:
            zsize = framemask.sum()
        else:
            framemask = np.ones(self.nfiles, dtype=bool)
            zsize = self.nfiles
        stack = np.zeros((zsize, ysize, xsize))

        if verbose:
            print('')
            print('median_combine: setting up stack for images')
            print('-------------------------------------------')
            print('Stack will have dimensions (%d, %d, %d)'
                  % (zsize, ysize, xsize))

        """ Loop over the frames to create the stack """
        count = 0
        for i in range(self.nfiles):
            if not framemask[i]:
                continue
            
            """ Process the data, possibly only partially, if desired """
            if usegain:
                gain = self.datainfo['gain'][i]
            else:
                gain = -1
            if usetexp:
                texp = self.datainfo['texp'][i]
            else:
                texp = -1

            tmp = self[i].process_data(bias=self.bias, gain=gain, texp=texp,
                                       flat=self.flat, trimsec=trimsec,
                                       verbose=verbose)

            """ Mask out the objects if use_objmask is set to True """
            if use_objmask:
                if self.objmasks is None:
                    raise ValueError('\n Run make_objmasks first\n ')
                tmp.apply_pixmask(self.objmasks[i])
                NaNmask = True

            """ Normalize if requested """
            if normalize is not None:
                mask = np.isfinite(tmp.data)
                normfac = tmp.normalize(method=normalize, mask=mask)
                if verbose:
                    print('    Normalizing by %f' % normfac)

            """ Set the sky to zero if requested """
            if zerosky is not None:
                skyval = tmp.sky_to_zero(zerosky)
                
            """ Put the processed data into the stack """
            stack[count] = tmp.data.copy()
            count += 1
            del tmp
        
        if verbose:
            print('')

        """ Actually form the median (or sum, if that was requested) """
        if method == 'sum':
            if NaNmask:
                if verbose:
                    print('median_combine: Computing summed frame using NaN'
                          ' masking')
                    print('    Can take a while...')
                outdat = np.nansum(stack, axis=0)
            else:
                if verbose:
                    print('median_combine: Computing summed frame (can take '
                          'a while)...')
                outdat = np.sum(stack, axis=0)
        else:
            if NaNmask:
                if verbose:
                    print('median_combine: Computing median frame using NaN '
                          'masking')
                    print('    Can take a while...')
                outdat = np.nanmedian(stack, axis=0)
            else:
                if verbose:
                    print('median_combine: Computing median frame (can take '
                          'a while)...')
                outdat = np.median(stack, axis=0)
        del stack

        """ Put the result into a HDU for saving or returning """
        phdu = pf.PrimaryHDU(outdat)
        if outobj is not None:
            phdu.header['object'] = outobj

        """ Write the output median file or return HDU """
        if outfile is not None:
            phdu.writeto(outfile, output_verify='ignore', overwrite=True)
            if verbose:
                print('    ... Wrote output to %s.' % outfile)
            return None
        else:
            return phdu
    
    # -----------------------------------------------------------------------

    def make_bias(self, outfile=None, trimsec=None, **kwargs):
        """ 

        This function median-combines the data to create a master dark/bias

        Optional inputs:
          outfile - output filename (default='Bias.fits')
          trimsec - a four-element list or array: [x1, y1, x2, y2] if something
                    smaller than the full frame is desired.  The coordinates
                    define the lower-left (x1, y1) and upper right (x2, y2)
                    corners of the trim section.

        """

        hdu = self.median_combine(outfile=outfile, trimsec=trimsec, **kwargs)

        if hdu is not None:
            return hdu

    # -----------------------------------------------------------------------

    def make_flat(self, outfile=None, biasfile=None, flatfile=None,
                  normalize='sigclip', trimsec=None, framemask=None,
                  use_objmask=False, **kwargs):
        """ 

        Combine the data in a way that is consistent with how you would make
         a flat-field frame
        NOTE: For making a sky flat from science data, see the make_skyflat
         method below

        Optional inputs:
          outfile      - output filename (default="Flat.fits")
          biasfile     - input bias file to subtract before combining. 
                          (default=None)
          gain         - gain factor to convert ADU to e-.  Default value
                          is None, since there is no advantage to converting
                          the flat-field frames to units of electrons
          normalize    - technique by which to normalize each frame before
                         combining.  Choices are:
                          'sigclip' - use clipped mean (default)
                          'median'  - use median
                          None      - no normalization
        """

        """  Call median_combine """
        hdu = self.median_combine(outfile=outfile, biasfile=biasfile,
                                  normalize=normalize, trimsec=trimsec,
                                  framemask=framemask, use_objmask=use_objmask,
                                  flatfile=flatfile, **kwargs)

        if hdu is not None:
            return hdu
    
    # -----------------------------------------------------------------------

    def apply_calib(self, outfiles=None, trimsec=None, biasfile=None,
                    usegain=False, flatfile=None, fringefile=None,
                    darkskyfile=None, zerosky=None, caldir=None, flip=None,
                    pixscale=0.0, rakey='ra', deckey='dec',
                    verbose=True):
        """

        Applies calibration corrections to the frames.
        All of the calibration steps are by default turned off (their
         associated keywords are set to None).
         To apply a particular calibration step, set the appropriate keyword.
         The possible steps, along with their keywords are:

          Keyword      Calibration step
          ----------  ----------------------------------
          biasfile    Bias subtraction
          flatfile    Flat-field correction
          fringefile  Fringe subtraction
          darkskyfile Dark-sky flat correction
          skysub      Subtract mean sky level if keyword set to True
          texp_key    Divide by exposure time (set keyword to fits header name)
          caldir      Directory containing the calibration files.  The default
                       value of None means that they are in the current
                       directory
          flip        None => no flip
          pixscale    If >0, apply a rough WCS using this pixel scale (RA and
                        Dec come from telescope pointing info in fits header)
          rakey       FITS header keyword for RA of telescope pointing.
                        Default = 'ra'
          deckey      FITS header keyword for Dec of telescope pointing.
                        Default = 'dec'
        
         Required inputs:

         Optional inputs (additional to those in the keyword list above):
          trimsec - a four-element list or array: [x1, y1, x2, y2] if something
                    smaller than the full frame is desired.  The coordinates
                    define the lower-left (x1, y1) and upper right (x2, y2)
                    corners of the trim section.
        
        """

        """ Read in calibration frames if they have been selected """
        self.load_calib(biasfile, flatfile, fringefile, darkskyfile,
                        caldir=caldir, verbose=verbose)

        """ Prepare to calibrate the data """
        if verbose:
            print('')
            print('Processing files...')
            print('-------------------')

        """ Loop through the frames, processing each one """
        outlist = []
        for i, hdu in enumerate(self):
            if usegain:
                gain = self.datainfo['gain'][i]
            else:
                gain = -1

            tmp = hdu.process_data(trimsec=trimsec, bias=self.bias,
                                   gain=gain, texp=self.datainfo['texp'][i],
                                   flat=self.flat, fringe=self.fringe,
                                   darkskyflat=self.darkskyflat,
                                   zerosky=zerosky, flip=flip, 
                                   pixscale=pixscale, rakey=rakey,
                                   deckey=deckey, verbose=verbose)
            if hdu.infile is not None:
                tmp.infile = hdu.infile

            if outfiles is not None:
                tmp.writeto(outfiles[i])
                print('   Wrote calibrated data to %s' % outfiles[i])
            else:
                outlist.append(tmp)
            print('')

        if outfiles is not None:
            return None
        else:
            return outlist

        # For LRIS B
        #  x1 = [400, 51, 51, 400]
        #  x2 = 1068
        #  y1 = 775
        #  y2 = 3200

    # -----------------------------------------------------------------------

    def make_objmasks(self, nsig=0.7, bpmlist=None, verbose=True):
        """

        Creates a list of object masks

        """

        objmasks = []

        for i, hdu in enumerate(self):
            if bpmlist is not None:
                bpm = bpmlist[i]
            else:
                bpm = None
            objmasks.append(hdu.make_objmask(nsig=nsig, bpm=bpm))
            if verbose and hdu.infile is not None:
                print(hdu.infile)

        self.objmasks = objmasks

    # -----------------------------------------------------------------------

    def make_skyflat(self, outfile='SkyFlat.fits', biasfile=None, flatfile=None,
                     normalize='sigclip', trimsec=None, **kwargs):
        """

        Creates a flat-field frame from the science exposures.  This is a
        one or three step process:
           1.  Make an initial flat.  If the object masks are already made,
               this is the last step, otherwise continue to step 2
           2.  Make object masks so that objects in the field do not contribute
               to the flat
           3.  Make the flat again, this time with the object masks

        """

        """ Make the first flat """
        if self.objmasks is not None:
            self.make_flat(outfile=outfile, biasfile=biasfile,
                           flatfile=flatfile, normalize=normalize,
                           trimsec=trimsec, **kwargs)
            return
        else:
            flat0 = 'FlatInit.fits'
            self.make_flat(outfile=flat0, biasfile=biasfile, flatfile=flatfile,
                           normalize=normalize, trimsec=trimsec, **kwargs)

        """ If the object masks don't exist then flat-field the data """
        print('')
        caldat = self.apply_calib(trimsec=trimsec, biasfile=biasfile,
                                  flatfile=flat0)
        orig = []
        for i, hdu in enumerate(self):
            orig.append(hdu.data.copy())
            hdu.data = caldat[i].data.copy()

        """ Make the object masks with the initially calibrated data """
        print('')
        print('Making object masks')
        self.make_objmasks()

        """
        Now reset the data, and then make a new flat but with the object
        masks this time
        """
        print('')
        print('Making final sky flat')
        for i, hdu in enumerate(self):
            hdu.data = orig[i].copy()
        self.make_flat(outfile=outfile, biasfile=biasfile, normalize=normalize,
                       trimsec=trimsec, **kwargs)

    # -----------------------------------------------------------------------

    def skysub_nir(self, biasfile=None, objmasks=None, ngroup=5,
                   outfiles=None, verbose=True):
        """

        Does the sky subtraction in the classic NIR imaging way, i.e., by
        creating a sky from the dithered observations that were taken
        immediately before or after the observation of interest.

        """

        """
        If there are object masks, then Mask the input data, saving the
        originals
        """
        orig = []
        if objmasks is not None:
            NaNmask = True
            for i, hdu in enumerate(self):
                orig.append(hdu.data.copy())
                hdu.apply_pixmask(objmasks[i], badval=1)
        else:
            NaNmask = False
            for hdu in self:
                orig.append(hdu.data.copy())
            
        """ Loop through the files """
        outlist = []
        dstep = int(floor((ngroup - 1) / 2.))
        for i, hdu in enumerate(self):
            if verbose:
                if hdu.infile is not None:
                    filename = hdu.infile
                else:
                    filename = 'File %d' % (i + 1)
                print('Sky subtraction for %s' % filename)
            # data = orig[i]
            if i < dstep:
                start = 0
            elif i > self.nfiles - ngroup:
                start = self.nfiles - ngroup
            else:
                start = max(0, i-dstep)
            end = min(start + ngroup, self.nfiles)
            indlist = np.arange(start, end).astype(int)
            mask = indlist != i
            framemask = np.zeros(self.nfiles, dtype=bool)
            framemask[indlist[mask]] = True
            # Original code below (normalized inputs rather than subtracting
            #    sky)
            # skyhdu = self.make_flat(outfile=None, biasfile=biasfile,
            #                         framemask=framemask, NaNmask=NaNmask)
            # skyhdu.data[~np.isfinite(skyhdu.data)] = 0.
            # scalefac = np.median(data) / np.median(skyhdu.data)
            # print('Scaling sky-flat data for %s by %f' %
            #       (hdu.infile, scalefac))
            skyhdu = self.median_combine(zerosky='sigclip', verbose=False,
                                         framemask=framemask, NaNmask=NaNmask,
                                         headverbose=False)
            hdu.sigma_clip()
            outdata = orig[i].data - hdu.mean_clip - skyhdu.data
            outlist.append(WcsHDU(outdata, hdu.header, verbose=False,
                           wcsverb=False))

        if outfiles is not None:
            for hdu, ofile in zip(outlist, outfiles):
                hdu.writeto(ofile)
                if verbose:
                    print('Wrote sky-subtracted data to %s' % ofile)
        return CCDSet(outlist, verbose=False)

    # -----------------------------------------------------------------------

    # def make_ones(self):

    # -----------------------------------------------------------------------

    def mark_crpix(self, flatfile=None, pixscale=None, fmin=1.,
                   fmax=10.):
        """

        Interactively sets (through clicking on a displayed image) the
        WCS reference pixel for each image, i.e., the CRPIX values.

        """

        """ Set up container for the CRPIX values"""
        crpix = Table(np.zeros((self.nfiles, 2)), names=['crpix1', 'crpix2'])
        notes = []

        """
        Set up the pixel scale to use
        The default is to use the WCS information in the file header, but if
        the pixscale parameter has been set then its value overrides any
        pixel scale information in the header
        """
        if pixscale is not None:
            pixscale /= 3600.

        """ Process the data if needed """
        if flatfile is not None:
            tmplist = self.apply_calib(flatfile=flatfile)
        else:
            tmplist = self
            
        """ Loop through the images, marking the object in each one """
        for im1, info in zip(tmplist, crpix):

            """ Open and display the image """
            dpar = DispParam(im1)
            dpar.display_setup(fmax=fmax, mode='xy', title=im1.infile)
            dispim = DispIm(im1)
            dispim.display(dpar=dpar)

            """ Run the interactive zooming and marking """
            dispim.start_interactive()
            plt.show()

            """ Set the crpix values to the marked location """
            hdr = im1.header
            if dispim.xmark is not None:
                info['crpix1'] = dispim.xmark + 1
                notes.append('..')
            elif 'CRPIX1' in hdr.keys():
                info['crpix1'] = hdr['crpix1']
                notes.append('Did not mark object')
            if dispim.ymark is not None:
                info['crpix2'] = dispim.ymark + 1
            elif 'CRPIX2' in hdr.keys():
                info['crpix2'] = hdr['crpix2']

        return crpix
    
    # -----------------------------------------------------------------------

    def update_refvals(self, crpix, crval):
        """

        Updates the headers to include new crpix and crval values

        """

        for hdu, pixval in zip(self, crpix):
            """
            If there is no WCS information in the input file, create a base
            version to be filled in later
            """
            if hdu.wcsinfo is None:
                hdu.wcsinfo = wcs.WCS(naxis=2)
                hdu.wcsinfo.wcs.ctype = ['RA---TAN', 'DEC--TAN']

            """ Update the CRPIX and CRVAL headers """
            hdu.crpix = (pixval['crpix1'], pixval['crpix2'])
            hdu.crval = crval
            # hdu.update_crval(crval, verbose=False)

    # -----------------------------------------------------------------------

    def update_wcshdr(self, hdrlist=None, wcslist=None, keeplist='all',
                      **kwargs):

        for i, hdu in enumerate(self):
            if hdrlist is not None:
                inhdr = hdrlist[i]
            else:
                inhdr = hdu.header
            if wcslist is not None:
                wcsinfo = wcslist[i]
            else:
                wcsinfo = hdu.wcsinfo
            outhdr = hdu.make_hdr_wcs(inhdr, wcsinfo, keeplist=keeplist,
                                      **kwargs)
            hdu.header = outhdr.copy()

    # -----------------------------------------------------------------------

    def align_crpix(self, radec=None, datasize=1500, fitsize=40, fwhmpix=10,
                    filtersize=5, savexc=False, verbose=True, debug=False,
                    **kwargs):
        """

        Uses the CRPIX values as the initial guesses for the shifts between
        the images, and then does a cross-correlation between the
        shifted images.  The cross-correlation uses data from each image
        that is centered at the requested (RA, Dec) location (assuming
        the WCS is correct) and uses data within a region of size
        datasize pixels centered on that location.  The results of the
        cross-correlation are used to update the CRPIX values.
        NOTE: These shifts are all in the native data frame and not in
        a WCS-aligned orientation.

        To summarize:
          1. Cut out data centered at the requested (RA, Dec) and with
             size datasize.  The default values, RA=None and Dec=None,
             will center the data regions at the location designated by
             the CRVAL keywords.
             NOTE: There should be an astronomical object at the requested
              (RA, Dec) location.
          2. Cross-correlate the data cutouts
          3. Fit a 2d Gaussian to the cross-correlation output
          4. Use the offset of the Gaussian's centroid from the center of the
             cross-correlated image to update the input CRPIX values.

        """

        if verbose:
            print('Refining CRPIX values (can take a while)')
            print('----------------------------------------')
            
        """
        Get the (x,y) position corresponding to the requested (RA, Dec)
        in the first image
        """
        hdu0 = self[0]
        hdr0 = self[0].header
        if radec is not None:
            ra = radec[0]
            dec = radec[1]
        else:
            ra = hdr0['crval1']
            dec = hdr0['crval2']
        xy0 = hdu0.wcsinfo.all_world2pix(ra, dec, 1)
        dcent0 = np.array([xy0[0], xy0[1]])
        if debug:
            print('   Requested center in image 0 at pix: %.2f %.2f'
                  % (dcent0[0], dcent0[1]))

        """ Set up container for old CRPIX values """
        ocrpix1 = np.zeros(self.nfiles)
        ocrpix2 = np.zeros(self.nfiles)

        """ Loop through the frames """
        for i, hdu in enumerate(self):
            if i == 0:
                ocrpix1[i] = hdu.header['crpix1']
                ocrpix2[i] = hdu.header['crpix2']
                continue
            
            """
            Get (x,y) position of requested (RA, Dec) and save original
            CRPIX values
            """
            xy = hdu.wcsinfo.all_world2pix(ra, dec, 1)
            dcent = np.array([xy[0], xy[1]])
            if debug:
                print('   Requested center in image %d at pix: %.2f %.2f'
                      % (i, dcent[0], dcent[1]))
            hdr = hdu.header
            ocrpix1[i] = hdr['crpix1']
            ocrpix2[i] = hdr['crpix2']

            """ Cross-correlate the data """
            if verbose:
                print('   Cross-correlating frames 0 and %d' % i)
            xccent = np.array((int(hdu.data.shape[1]/2.),
                               int(hdu.data.shape[0]/2.)))
            xc = hdu0.cross_correlate(hdu, datacent=xccent, othercent=xccent,
                                      datasize=datasize, **kwargs)
            if savexc:
                outfile = 'xc%d.fits' % i
                xc.writeto(outfile)
                if verbose:
                    print('   Saved cross-correlation image to %s' % outfile)

            """
            Do a small median smoothing on the cross-correlated image to
            get rid of possible cosmic-ray / bad-pixel overlaps
            """
            if debug:
                tmpdat = xc.data
                yxpeak = np.unravel_index(tmpdat.argmax(), tmpdat.shape)
                print('   Peak in xcorr before smoothing: %d    %d' %
                      (yxpeak[1], yxpeak[0]))
            xc.data = filters.median_filter(xc.data, size=filtersize)
            if debug:
                tmpdat = xc.data
                yxpeak = np.unravel_index(tmpdat.argmax(), tmpdat.shape)
                print('   Peak in xcorr after smoothing:  %d    %d' %
                      (yxpeak[1], yxpeak[0]))
                xc.writeto('xctest.fits')

            """
            If the WCS is basically correct, then there should be a
             peak in the cross-correlation image pretty close to the
             position derived from taking the difference in the CRPIX
             values.  Therefore fit to peak within a small box centered
             at this position.
            The cross_correlate method would produce a peak at the center of
             the cross correlation image if there were no shift between the
             images.  Therefore, the offset has to be applied from the center
             of the cross correlation image
            """
            dposcr = dcent - dcent0
            x0 = int(xc.data.shape[1]/2.) + dposcr[0]
            y0 = int(xc.data.shape[0]/2.) + dposcr[1]
            dx = int(fitsize / 2.)
            xmin = int(x0 - dx)
            xmax = int(xmin + fitsize)
            ymin = int(y0 - dx)
            ymax = int(ymin + fitsize)
            data = xc.data[ymin:ymax, xmin:xmax]
            if debug:
                print('   Expected peak position:         %.2f %.2f' % (x0, y0))
                print('   xrange = [%d:%d]' % (xmin, xmax))
                print('   yrange = [%d:%d]' % (ymin, ymax))

            """ Fit to the cross-correlation peak """
            if verbose:
                print('   Fitting to cross-correlation peak')
            fit = imfit.ImFit(data)
            yguess, xguess = np.unravel_index(data.argmax(), data.shape)
            mod = fit.gaussians(xguess, yguess, fwhmpix=fwhmpix, dxymax=5,
                                fitbkgd=False, verbose=False,
                                usemoments=False)
            xfit = mod.x_mean + xmin
            yfit = mod.y_mean + ymin
            if debug:
                print('   Found xcorr peak in cutout: %.2f %.2f' %
                      (mod.x_mean.value, mod.y_mean.value))
                print('   ==> peak in full xcorr frame: %.2f %.2f' %
                      (xfit, yfit))

            """
            Determine if any adjustments to the CRPIX values are needed
            The cross-correlation peak will be offset from the _center_ of the
             cross correlation image
            """
            dxxc = xfit - (xc.shape[1]/2.)
            dyxc = yfit - (xc.shape[0]/2.)
            # print('%.2f %.2f   %.2f %.2f   %.2f %.2f' %
            #       (dxxc, dyxc, dposcr[0], dposcr[1], (dxxc-dposcr[0]),
            #        (dyxc-dposcr[1])))

            """ Adjust the CRPIX values """
            newpix1 = hdr0['crpix1'] + dxxc
            newpix2 = hdr0['crpix2'] + dyxc
            hdr['ocrpix1'] = hdr['crpix1']
            hdr['ocrpix2'] = hdr['crpix2']
            hdu.crpix = [newpix1, newpix2]

            """ Clean up """
            del xc
            if debug:
                print('')

        """ Report on updated values if requested"""
        if verbose:
            print('')
            print(' n  CRPIX1_0  CRPIX1     dx      CRPIX2_0  CRPIX2     dy')
            print('--- -------- --------  ------    -------- --------  ------')
            count = 0
            for pix1, pix2, hdu in zip(ocrpix1, ocrpix2, self):
                crpix1 = hdu.header['crpix1']
                crpix2 = hdu.header['crpix2']
                dx = crpix1 - pix1
                dy = crpix2 - pix2
                print('%2d  %8.2f %8.2f  %+6.2f    %8.2f %8.2f  %+6.2f' %
                      (count, pix1, crpix1, dx, pix2, crpix2, dy))
                count += 1

    # -----------------------------------------------------------------------

    def plot_panel(self, hdu, plotpars, ax=None, mode='radec', axlabel=False,
                   fontsize=None, verbose=True,
                   **kwargs):
        """

        Plots a single panel

        """

        """ If no axis has been defined, just do a single-panel plot"""
        if ax is None:
            fig = plt.figure()
            ax = plt.gca()

        """ Select the portion of the image to be displayed """
        if mode == 'xy' or mode == 'pix':
            imcent = (plotpars['x_cent'], plotpars['y_cent'])
            x1, y1, x2, y2 = hdu.subim_bounds_xy(imcent, plotpars['imsize'])
            plthdu = hdu.cutout_xy(x1, y1, x2, y2, verbose=verbose)

        else:
            imcent = (plotpars['ra'], plotpars['dec'])
            plthdu = hdu.cutout_radec(imcent, plotpars['imsize'],
                                      verbose=verbose)

        """ Set up the default display parameters"""
        dpar = DispParam(plthdu)

        """ Override and/or add to default parameters if requested """
        for k in plotpars.keys():
            dpar[k] = plotpars[k]

        """ Set up internal labels """
        # for k in ['tltext', 'tctext', 'trtext', 'bltext', 'bctext', 'brtext']:
        #     if k in plotpars.keys():
        #         dpar[k] = plotpars[k]

        if 'fmax' in plotpars.keys():
            dpar.display_setup(mode=mode, fmax=plotpars['fmax'],
                               verbose=verbose, **kwargs)
        else:
            dpar.display_setup(mode=mode, verbose=verbose, **kwargs)

        """ Update the display parameters """
        dpar.axlab = 'off'

        """ Actually display the image """
        dispim = DispIm(plthdu)
        dispim.display(ax=ax, axlabel=axlabel, fontsize=fontsize,
                       mode=mode, dpar=dpar)

    # -----------------------------------------------------------------------

    def plot_multipanel(self, plotinfo, mode='radec', ncol=None, maxrows=None,
                        outfile=None, panelsize=2.0, debug=False, **kwargs):
        """

        Makes a multiple-panel plot using the image data in the object.

        Required inputs:
          plotinfo  -  Either an astropy Table object or a list of dicts that
                       contains information that governs the plotting of
                       each panel.
                       Required columns/keys in plotinfo:
                         x_cent or RA, depending if mode is 'xy' or 'radec'
                         y_cent or Dec, depending if mode is 'xy' or 'radec'
                         imsize - size is in pixels if mode is 'xy' or arcsec
                                   if mod is 'radec'

        """

        """ First make sure that plotinfo is the correct data type and size """
        isdlist = False
        if isinstance(plotinfo, (Table, list)):
            if len(plotinfo) != len(self):
                raise IndexError('\nLength of plotinfo does not match number'
                                 ' of images\n')
            if isinstance(plotinfo, list):
                if isinstance(plotinfo[0], dict):
                    isdlist = True
                else:
                    raise TypeError('\nIf plotinfo is a list, then it must be '
                                    '  list of dict objects\n')
        else:
            raise TypeError('\nplotinfo parameter must be an astropy Table '
                            'or a list of dicts\n')

        """ Make sure the image center and image size keys are in plotinfo """
        if isdlist:
            keylist = plotinfo[0].keys()
        else:
            keylist = plotinfo.keys()
        if mode == 'xy':
            critkeys = ['x_cent', 'y_cent', 'imsize']
        else:
            critkeys = ['ra', 'dec', 'imsize']
        for k in critkeys:
            if k not in keylist:
                raise KeyError('\nplotinfo is missing %s\n' % k)

        """ Determine the number of columns and rows for the multipanel plot """
        if ncol is None:
            nn = sqrt(len(self))
            if len(self) % nn == 0:
                ncol = int(nn)
            else:
                ncol = int(nn) + 1
        if len(self) % ncol == 0:
            nrow = int(len(self) / ncol)
        else:
            nrow = int(len(self) / ncol) + 1
        if debug:
            print('Number of rows x columns = %d x %d' % (nrow, ncol))
            print('')

        """ Set up plotting basics """
        if maxrows is not None:
            if nrow > maxrows:
                figsize = (ncol * panelsize, maxrows * panelsize)
                totfigs = int(nrow / maxrows) + 1
                lastsize = (ncol * panelsize, (nrow % maxrows) * panelsize)
            ny = min(nrow, maxrows)
        else:
            figsize = (ncol * panelsize, nrow * panelsize)
            totfigs = 1
            ny = nrow
        xsize = 1.0 / ncol - 0.005
        ysize = 1.0 / ny - 0.005

        """ Loop through the images, plotting each one """
        fig, axes = plt.subplots(ny, ncol, figsize=figsize)
        if debug:
            print(axes.shape)
        row = -1
        # for i, hdu in enumerate(self):
        for i in range(ny * ncol):
            col = i % ncol
            if col == 0:
                row += 1
            if debug:
                print(i, row, col)
            if ny > 1:
                ax = axes[row, col]
            else:
                ax = axes[col]
            if i < len(self):
                print(self[i].infile)
                self.plot_panel(self[i], plotinfo[i], ax=ax, mode=mode,
                                axlabel=False, **kwargs)
            else:
                ax.set_axis_off()
        plt.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        if outfile is not None:
            plt.savefig(outfile)
            print('')
            print('Saved figure to %s' % outfile)
            print('')
        else:
            plt.show()

    # -----------------------------------------------------------------------

    def check_alignment(self, ra, dec, imsize, fmax=20., **kwargs):
        """

        Use this method if the CCDSet object is multiple exposures of the
        same field.  The method will plot cutouts from each exposure,
        centered at the same (RA, Dec) position, according to the WCS
        information in that exposure.  This allows the user to see if the
        exposures are astrometrically aligned or not

        """

        """
        Make a list of dict objects that contains information needed for
        the plotting
        """
        plotinfo = []
        for hdu, info in zip(self, self.datainfo):
            pdict = {'ra': ra, 'dec': dec, 'imsize': imsize, 'fmax': fmax}
            pdict['crosshair'] = (0, 0)
            if info['basename'] is not None:
                pdict['tltext'] = info['basename']
            plotinfo.append(pdict)

        """ Make the multipanel plot """
        self.plot_multipanel(plotinfo, **kwargs)

    # -----------------------------------------------------------------------

    @staticmethod
    def _make_swarp_keepflag(keylist=None):
        """

        Set a default list of keywords to keep in the proper format for
        swarp

        """

        if keylist is None:
            keepkeys = ['object', 'telescop', 'instrume', 'filter']
        else:
            keepkeys = keylist
        for i, k in enumerate(keepkeys):
            if i == 0:
                keys = k.upper()
            else:
                keys = '%s,%s' % (keys, k.upper())
        keyflag = '-COPY_KEYWORDS %s ' % keys

        return keyflag

    # -----------------------------------------------------------------------

    @staticmethod
    def _set_default_swarppars():
        """

        Sets the default values for parameters to be passed to swarp.
        These default values can be overridden by values in the swarppars
        parameter that is passed to run_swarp

        """

        swarppars = {
            'combtype': 'median',
            'weighttype': 'NONE',
            'keepkeys': ['object', 'telescop', 'instrume', 'filter']
        }

        return swarppars

    # -----------------------------------------------------------------------

    def _make_swarp_addstr(self, swarppars):
        """

        Makes a string, in the appropriate format, containing optional
        parameters to be passed to swarp in the run_swarp method

        """

        """ First set the default values """
        pars = self._set_default_swarppars()
        if swarppars is not None:
            for k in swarppars.keys():
                pars[k] = swarppars[k]
                print('Setting pars[%s] to %s' % (k, swarppars[k]))

        """ Loop through the pars dictionary and create the output string """
        addstr = ' '
        for k in pars.keys():
            if k == 'combtype':
                addstr += '-COMBINE_TYPE %s ' % pars[k].upper()
            elif k == 'keepkeys':
                addstr += self._make_swarp_keepflag(pars[k])

        return addstr

    # -----------------------------------------------------------------------

    def run_swarp(self, config, outfile, swarppars=None, whtflags=None,
                  flags=None, outwhtsuff='wht', verbose=True):
        """

        Runs swarp on the files in the CCDSet object, using the provided
        swarp configuration file (config parameter)

        """

        """ Create input filelist in swarp format """
        for i, info in enumerate(self):
            if i == 0:
                infiles = '%s' % info.infile
            else:
                infiles += ',%s' % info.infile

        """ Set up the base call to swarp """
        swarpcommand = 'swarp %s -c %s ' % (infiles, config)
        if outfile is not None:
            swarpcommand += '-IMAGEOUT_NAME %s ' % outfile
            if outwhtsuff is not None:
                outwht = outfile.replace('.fits', '_%s.fits' % outwhtsuff)
                swarpcommand += '-WEIGHTOUT_NAME %s ' % outwht

        """ Add additional swarp flags """
        swarpcommand += '%s ' % self._make_swarp_addstr(swarppars)

        print('')
        print(swarpcommand)

    # -----------------------------------------------------------------------

    def fit_4qso(self, infile, outfile, reflab, fittype='moffat',
                 lab=('A', 'B', 'C', 'D')):
        """

        Fits four two-dimensional PSFs to the data in an image, guided
        by the initial guesses that are contained in the input file (passed
        via the infile parameter).

        """

        """ Load the initial guesses for the positions into a Table """
        posnames = ['xA', 'yA', 'xB', 'yB', 'xC', 'yC', 'xD', 'yD']
        colnames = ['fname'] + posnames
        inittab = Table.read(infile, format='ascii.no_header', names=colnames)
        if len(inittab) != len(self):
            raise IndexError('Input table must match number of files')

        """ Convert the the Table data into an array for ease of use """
        N = len(self)
        indat = ((inittab[posnames]).as_array().view(float)).reshape((N, 4, 2))

        """ Set up container for output positions """
        outtab = inittab.copy()
        for key in posnames:
            outtab[key] = 0.

        """ Loop over the files """
        count = 0
        for im, info, out in zip(self, inittab, outtab):
            """ Create an ImFit object """
            qsofit = imfit.ImFit(im.data)

            """ Set up the initial guesses in the proper format """
            initpos = Table(indat[count, :, :], names=['x', 'y'])
            initpos['lab'] = lab

            """ Do the fitting """
            print('')
            print('Fitting model for file %s.  Be patient' % info['fname'])
            mod = qsofit.moffats(initpos['x'], initpos['y'], fwhmpix=9.)

            """ Store the model positions """
            outpos = initpos.copy()
            for i in range(4):
                outpos['x'][i] = mod[i].x_0.value
                outpos['y'][i] = mod[i].y_0.value

            """ Compute the positions relative to the reference image """
            mask = initpos['lab'] == reflab
            x0 = outpos['x'][mask][0]
            y0 = outpos['y'][mask][0]
            outpos['x'] = x0 - outpos['x']
            outpos['y'] -= y0

            """ Convert pix to arcsec, if the image has WCS info """
            if im.pixscale is not None:
                outpos['x'] *= im.pixscale[0]
                outpos['y'] *= im.pixscale[1]
            print(outpos)

            """ Put the fitted positions into the output table """
            tmp = np.zeros((4, 2))
            tmp[:, 0] = outpos['x']
            tmp[:, 1] = outpos['y']
            out[posnames] = tmp.flatten()

            """ Make diagnostic fits file """
            moddat = mod(qsofit.x, qsofit.y) + qsofit.mock_noise()
            modim = WcsHDU(moddat, wcsverb=False)
            modim.wcsinfo = im.wcsinfo.deepcopy()
            modim.save('modim_%02d.fits' % count)
            diffdat = im.data - modim.data
            diff = WcsHDU(diffdat, wcsverb=False)
            diff.wcsinfo = im.wcsinfo.deepcopy()
            diff.save('resid_%02d.fits' % count)

            count += 1
        print('')
        outtab.write(outfile, format='ascii.basic')
