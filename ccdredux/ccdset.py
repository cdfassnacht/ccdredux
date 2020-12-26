"""

Defines a CCDSet class that can be used for the standard calibration steps
for CCD and similar data sets
"""

import os
import sys
import numpy as np
from math import floor

from astropy.io import fits as pf

from specim.imfuncs import WcsHDU

pyversion = sys.version_info.major


# ===========================================================================

class CCDSet(list):
    """

    A class that can be used to perform standard calibration steps for
    a collection of CCD and CCD-like data sets

    """

    def __init__(self, inlist, hext=0, wcsext=None, texpkey=None, gainkey=None,
                 rdnoisekey=None, coaddkey=None, filtkey=None, wcsverb=False,
                 verbose=True, **kwargs):
        """

        Instantiates a CCDSet object by reading in the relevant data sets

        """

        """ Set up the empty CCDSet container by calling the superclass """
        if pyversion == 2:
            super(Image, self).__init__()
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
        self.texp = np.ones(self.nfiles)
        self.gain = -1. * np.ones(self.nfiles)
        self.bias = None
        self.flat = None
        self.fringe = None
        self.darkskyflat = None
        
        """ Load the data """
        for i, f in enumerate(inlist):
            tmp = WcsHDU(f, hext=hext, wcsext=wcsext, verbose=False,
                         wcsverb=False, **kwargs)
            if texpkey is not None:
                if texpkey.upper() in tmp.header.keys():
                    self.texp[i] = tmp.header[texpkey]
                else:
                    print('WARNING: %s keyword not found in header of file %s' %
                          (texpkey, tmp.infile))
            if gainkey is not None:
                if gainkey.upper() in tmp.header.keys():
                    self.gain[i] = tmp.header[gainkey]
                else:
                    print('WARNING: %s keyword not found in header of file %s' %
                          (gainkey, tmp.infile))
            self.append(tmp)

        """ Summarize the inputs """
        if verbose:
            self.print_summary()
            
    # -----------------------------------------------------------------------

    def print_summary(self):
        """

        Summarizes the input file characteristics

        """

        print('File            Object          texp(s) gain ')
        print('--------------- --------------- ------- -----')
        for hdu, texp, gain in zip(self, self.texp, self.gain):
            infile = os.path.basename(hdu.infile)
            if infile[-5:] == '.fits':
                infile = infile[:-5]
            if 'OBJECT' in hdu.header.keys():
                obj = hdu.header['object']
            else:
                obj = 'N/A'
            print('%-15s %-15s %7.2f %5.2f' % (infile, obj, texp, gain))
        return
    
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

    def median_combine(self, outfile=None, method='median', goodmask=None,
                       trimsec=None, biasfile=None, flatfile=None, gain=None,
                       normalize=None, zerosky=None, NaNmask=False,
                       verbose=True):
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
        self.load_calib(biasfile, flatfile)

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
        if goodmask is not None:
            zsize = goodmask.sum()
        else:
            goodmask = np.ones(self.files, dtype=bool)
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
            if not goodmask[i]:
                continue
            
            if verbose:
                print(' %s' % self[i].infile)

            """ Process the data (bias and gain only), if desired """
            tmp = self[i].process_data(gain=gain[i], trimsec=trimsec)

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
            del(tmp)
        
        print('')

        """ Actually form the median (or sum, if that was requested) """
        if method == 'sum':
            if NaNmask:
                print('median_combine: Computing summed frame using NaN'
                      ' masking')
                print('    Can take a while...')
                outdat = np.nansum(stack, axis=0)
            else:
                print('median_combine: Computing summed frame (can take '
                      'a while)...')
                outdat = np.sum(stack, axis=0)
        else:
            if NaNmask:
                print('median_combine: Computing median frame using NaN '
                      'masking')
                print('    Can take a while...')
                outdat = np.nanmedian(stack, axis=0)
            else:
                print('median_combine: Computing median frame (can take '
                      'a while)...')
                outdat = np.median(stack, axis=0)
        del stack

        """ Put the result into a HDU for saving or returning """
        phdu = pf.PrimaryHDU(outdat)

        """ Write the output median file or return HDU """
        if outfile is not None:
            phdu.writeto(outfile, output_verify='ignore', overwrite=True)
            print('    ... Wrote output to %s.' % outfile)
            return None
        else:
            return phdu
    
    # -----------------------------------------------------------------------

    def make_bias(self, outfile='Bias.fits', trimsec=None, gain=None,
                  **kwargs):
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

    def make_flat(self, outfile='Flat.fits', biasfile=None, gain=None, 
                  normalize='sigclip', trimsec=None, goodmask=None,
                  **kwargs):
        """ 

        Combine the data in a way that is consistent with how you would make
        a flat-field frame


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
                                  gain=gain, normalize=normalize,
                                  trimsec=trimsec,
                                  goodmask=goodmask, **kwargs)

        if hdu is not None:
            return hdu
    
    # -----------------------------------------------------------------------

    def apply_calib(self, outfiles=None, trimsec=None,
                    biasfile=None, flatfile=None, fringefile=None,
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
        self.load_calib(biasfile, flatfile, fringefile, darkskyfile)

        """ Prepare to calibrate the data """
        if verbose:
            print('')
            print('Processing files...')
            print('-------------------')

        """ Loop through the frames, processing each one """
        outlist = []
        for i, hdu in enumerate(self):
            tmp = hdu.process_data(trimsec=trimsec, bias=self.bias,
                                   gain=self.gain[i], texp=self.texp[i],
                                   flat=self.flat, fringe=self.fringe,
                                   darkskyflat=self.darkskyflat,
                                   zerosky=zerosky, flip=flip, 
                                   pixscale=pixscale, rakey=rakey,
                                   deckey=deckey, verbose=verbose)

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

    def skysub_nir(self, biasfile=None, objmasklist=None, ngroup=5):
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
        if objmasklist is not None:
            NaNmask = True
            for i, hdu in enumerate(self):
                orig.append(hdu.data.copy())
                hdu.apply_pixmask(objmasklist[i], badval=1)
        else:
            NaNmask = False
            for hdu in self:
                orig.append(hdu.data.copy())
            
        """ Loop through the files """
        outlist = []
        dstep = int(floor((ngroup - 1) / 2.))
        for i, hdu in enumerate(self):
            data = orig[i]
            if i < dstep:
                start = 0
            elif i > self.nfiles - ngroup:
                start = self.nfiles - ngroup
            else:
                start = max(0, i-dstep)
            end = min(start + ngroup, self.nfiles)
            indlist = np.arange(start, end).astype(int)
            mask = indlist != i
            goodmask = np.zeros(self.nfiles, dtype=bool)
            goodmask[indlist[mask]] = True
            # Code below was original
            # skyhdu = self.make_flat(outfile=None, biasfile=biasfile,
            #                         goodmask=goodmask, NaNmask=NaNmask)
            # skyhdu.data[~np.isfinite(skyhdu.data)] = 0.
            # scalefac = np.median(data) / np.median(skyhdu.data)
            # print('Scaling sky-flat data for %s by %f' %
            #       (hdu.infile, scalefac))
            skyhdu = self.median_combine(zerosky='sigclip',
                                         goodmask=goodmask, NaNmask=NaNmask)
            hdu.sigma_clip()
            outdata = orig[i].data - hdu.mean_clip - skyhdu.data
            outlist.append(pf.PrimaryHDU(outdata, hdu.header))

        return CCDSet(outlist)

    # -----------------------------------------------------------------------

    def make_objmasks(self, nsig=1.5, bpmlist=None):
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

        return objmasks

