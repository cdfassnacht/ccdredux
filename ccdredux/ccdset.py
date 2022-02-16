"""

Defines a CCDSet class that can be used for the standard calibration steps
for CCD and similar data sets
"""

import sys
from os import path
import numpy as np
from math import floor

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

    def __init__(self, inlist, hext=0, wcsext=None, filecol='infile',
                 tabformat=None, infokeys=None, texpkey=None, gainkey=None,
                 wcsverb=False, verbose=True, **kwargs):
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
        self.flat = None
        self.fringe = None
        self.darkskyflat = None
        self.objmasks = None
        
        """ Set up for loading the data """
        if isinstance(inlist, (list, tuple)):
            self.datainfo = Table()
            if isinstance(inlist[0], str):
                self.datainfo['infile'] = inlist
            elif isinstance(inlist[0],
                            (pf.PrimaryHDU, pf.ImageHDU, WcsHDU, Image)):
                filelist = []
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

        """ Load the data into the object """
        if verbose:
            print('')
            print('Loading data...')
        for f, info in zip(inlist, self.datainfo):
            if isinstance(f, (pf.PrimaryHDU, pf.ImageHDU, WcsHDU, Image)):
                infile = f
                if f.infile is not None:
                    inbase = path.basename(f.infile)
                else:
                    inbase = None
            else:
                infile = info['infile']
                inbase = path.basename(infile)
            tmp = WcsHDU(infile, hext=hext, wcsext=wcsext, verbose=False,
                         wcsverb=wcsverb, **kwargs)
            if inbase is not None:
                tmp.infile = inbase
                info['infile'] = inbase
            self.append(tmp)

        """ Put the requested information into datainfo table """
        keylist = ['object']
        if texpkey is not None:
            keylist.append(texpkey)
        if gainkey is not None:
            keylist.append(gainkey)
        if infokeys is not None:
            for key in infokeys:
                if key.lower() != 'object':
                    keylist.append(key)
        self.read_infokeys(keylist)

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

        """ Summarize the inputs """
        if verbose:
            print('')
            self.print_summary(keylist)
            
    # -----------------------------------------------------------------------

    def read_infokeys(self, infokeys):
        """

        Adds information that is designated by the passed keywords to the
        datainfo table

        """

        """ Start by adding appropriate columns to the table """
        for key in infokeys:
            self.datainfo[key] = None

        """ Get the information from the fits headers, if available """
        for hdu, info in zip(self, self.datainfo):
            hdr = hdu.header
            for key in infokeys:
                if key.upper() in hdr.keys():
                    info[key] = hdr[key.upper()]
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

        sumkeys = ['infile']
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
              '  CRPIX2 ')
        print('------------------------ ----------- ----------- --------'
              ' --------')
        for i, hdu in enumerate(self):
            hdr = hdu.header
            if hdu.infile is not None:
                f = hdu.infile[:-5]
            else:
                f = 'Image %d' % i
            print('%-24s %11.7f %+11.7f %8.2f %8.2f' %
                  (f, hdr['crval1'], hdr['crval2'], hdr['crpix1'],
                   hdr['crpix2']))
            del hdr

    # -----------------------------------------------------------------------

    def read_calfile(self, filename, file_description, hext=0, verbose=True):
        """

        Reads in a calibration file

        """
        if verbose:
            print('Reading %s file: %s' % (file_description, filename))
            
        try:
            calhdu = WcsHDU(filename, hext=hext, verbose=False, wcsverb=False)
        except FileNotFoundError:
            print(' ERROR: Requested %s file %s does not exist' % 
                  (file_description, filename))
            print('')
            return None
        except OSError:
            print(' ERROR reading file %s' % filename)
            return None
        return calhdu

    # -----------------------------------------------------------------------

    def load_calib(self, biasfile=None, flatfile=None, fringefile=None,
                   darkskyfile=None, hext=None, verbose=True):
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
        if verbose:
            print('Loading any requested calibration files')
            print('---------------------------------------')
        if biasfile is not None:
            self.bias = self.read_calfile(biasfile, 'bias/dark', hext=hext,
                                          verbose=verbose)
            if self.bias is None:
                raise OSError('Error reading %s' % biasfile)

        if flatfile is not None:
            self.flat = self.read_calfile(flatfile, 'flat-field', hext=hext)
            if self.flat is None:
                raise OSError('Error reading %s' % flatfile)

        if fringefile is not None:
            self.fringe = self.read_calfile(fringefile, 'fringe', hext=hext)
            if self.fringe is None:
                raise OSError('Error reading %s' % fringefile)

        if darkskyfile is not None:
            self.darkskyflat = self.read_calfile(darkskyfile, 'dark-sky flat',
                                                 hext=hext)
            if self.darkskyflat is None:
                raise OSError('Error reading %s' % darkskyfile)

    # -----------------------------------------------------------------------

    def median_combine(self, outfile=None, method='median', framemask=None,
                       trimsec=None, biasfile=None, flatfile=None,
                       usegain=False, normalize=None, zerosky=None,
                       use_objmask=False, NaNmask=False, verbose=True):
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
        self.load_calib(biasfile, flatfile, verbose=verbose)

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
            
            """ Process the data (bias and gain only), if desired """
            if usegain:
                gain = self.datainfo['gain'][i]
            else:
                gain = -1
            tmp = self[i].process_data(bias=self.bias, gain=gain,
                                       trimsec=trimsec, verbose=verbose)

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

    def make_flat(self, outfile=None, biasfile=None, normalize='sigclip',
                  trimsec=None, framemask=None, use_objmask=False, **kwargs):
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
                                  **kwargs)

        if hdu is not None:
            return hdu
    
    # -----------------------------------------------------------------------

    def apply_calib(self, outfiles=None, trimsec=None, biasfile=None,
                    usegain=False, flatfile=None, fringefile=None,
                    darkskyfile=None, zerosky=None, flip=None,
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
                        verbose=verbose)

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

    def make_objmasks(self, nsig=0.7, bpmlist=None):
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

        self.objmasks = objmasks

    # -----------------------------------------------------------------------

    def make_skyflat(self, outfile='SkyFlat.fits', biasfile=None,
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
                           normalize=normalize, trimsec=trimsec, **kwargs)
            return
        else:
            flat0 = 'FlatInit.fits'
            self.make_flat(outfile=flat0, biasfile=biasfile,
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
                                         framemask=framemask, NaNmask=NaNmask)
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

    def mark_crpix(self, flatfile=None, pixscale=None, fmin=1.,
                   fmax=10.):
        """

        Interactively sets (through clicking on a displayed image) the
        WCS reference pixel for each image, i.e., the CRPIX values.

        """

        """ Set up container for the CRPIX values"""
        crpix = Table(np.zeros((self.nfiles, 2)), names=['crpix1', 'crpix2'])

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
            if dispim.xmark is not None:
                info['crpix1'] = dispim.xmark + 1
            if dispim.ymark is not None:
                info['crpix2'] = dispim.ymark + 1

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
            hdu.update_crval(crval, verbose=False)

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
                    filtersize=5, savexc=False, verbose=True, **kwargs):
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
            xc.data = filters.median_filter(xc.data, size=filtersize)

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
            x0 = (xc.data.shape[1]/2.) + dposcr[0]
            y0 = (xc.data.shape[0]/2.) + dposcr[1]
            dx = int(fitsize / 2.)
            xmin = int(x0 - dx)
            xmax = int(xmin + fitsize)
            ymin = int(y0 - dx)
            ymax = int(ymin + fitsize)
            data = xc.data[ymin:ymax, xmin:xmax]
            
            """ Fit to the cross-correlation peak """
            if verbose:
                print('   Fitting to cross-correlation peak')
            fit = imfit.ImFit(data)
            mod = fit.gaussians(dx, dx, fwhmpix=fwhmpix,
                                fitbkgd=True, verbose=False,
                                usemoments=False)
            xfit = mod.x_mean + xmin
            yfit = mod.y_mean + ymin

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
