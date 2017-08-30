from __future__ import print_function

try:
    xrange = xrange
    # We have Python 2
except:
    xrange = range
    # We have Python 3

import copy
import numpy
from numpy import array, zeros, exp, log10, log, dot, sqrt, diag
from . import fastmath
from .jacobian import Jacobian, UnitJacobian
from .shape import Shape, g1g2_to_e1e2, e1e2_to_g1g2

from . import moments

from .gexceptions import GMixRangeError, GMixFatalError

from .sersicfits import (
    _sersic_data_5gauss,
    _sersic_data_10gauss  
)

from . import _gmix

import scipy.interpolate

def make_gmix_model(pars, model):
    """
    get a gaussian mixture model for the given model
    """
    model_num=get_model_num(model)
    if model==GMIX_COELLIP:
        return GMixCoellip(pars)

    elif model==GMIX_SERSIC:
        return GMixSersic(pars)
    elif model==GMIX_SERSIC5:
        return GMixSersic5(pars)
    elif model==GMIX_SERSIC10:
        return GMixSersic10(pars)

    elif model==GMIX_FULL:
        return GMix(pars=pars)
    else:
        return GMixModel(pars, model)

class GMix(object):
    """
    A general two-dimensional gaussian mixture.

    parameters
    ----------
    Send either ngauss= or pars=

    ngauss: number, optional
        number of gaussians.  data will be zeroed
    pars: array-like, optional
        6*ngauss elements to fill the gaussian mixture.

    methods
    -------
    copy(self):
        make a new copy of this GMix
    convolve(psf):
        Get a new GMix that is the convolution of the GMix with the input psf
    get_T():
        get T=sum(p*T_i)/sum(p)
    get_sigma():
        get sigma=sqrt(T/2)
    get_psum():
        get sum(p)
    set_psum(psum):
        set new overall sum(p)
    get_cen():
        get cen=sum(p*cen_i)/sum(p)
    set_cen(row,col):
        set the overall center to the input.
    """
    def __init__(self, ngauss=None, pars=None):

        self._model      = GMIX_FULL
        self._model_name = 'full'

        if ngauss is None and pars is None:
            raise GMixFatalError("send ngauss= or pars=")

        if pars is not None:
            npars = len(pars)
            if (npars % 6) != 0:
                raise GMixFatalError("len(pars) must be mutiple of 6 "
                                     "got %s" % npars)
            self._ngauss=npars//6
            self.reset()
            self.fill(pars)
        else:
            self._ngauss=ngauss
            self.reset()
        
        self._set_f8_type()

    def _set_f8_type(self):
        tmp=numpy.zeros(1)
        self._f8_type=tmp.dtype.descr[0][1]

    def get_data(self):
        """
        Get the underlying array
        """
        return self._data

    def get_full_pars(self):
        """
        Get a full parameter description.
           [p1,row1,col1,irr1,irc1,icc1,
            p2,row2,col2,irr2,irc2,icc2,
            ...
           ]

        """

        gm=self._get_gmix_data()

        n=self._ngauss
        pars=numpy.zeros(n*6)
        beg=0
        for i in xrange(n):
            pars[beg+0] = gm['p'][i]
            pars[beg+1] = gm['row'][i]
            pars[beg+2] = gm['col'][i]
            pars[beg+3] = gm['irr'][i]
            pars[beg+4] = gm['irc'][i]
            pars[beg+5] = gm['icc'][i]
            
            beg += 6
        return pars

    def get_cen(self):
        """
        get the center position (row,col)
        """

        gm=self._get_gmix_data()
        psum=gm['p'].sum()
        rowsum=(gm['row']*gm['p']).sum()
        colsum=(gm['col']*gm['p']).sum()

        row=rowsum/psum
        col=colsum/psum

        return row,col
    
    def set_cen(self, row, col):
        """
        Move the mixture to a new center
        """
        gm=self._get_gmix_data()

        row0,col0 = self.get_cen()
        row_shift = row - row0
        col_shift = col - col0

        gm['row'] += row_shift
        gm['col'] += col_shift

    def get_T(self):
        """
        get weighted average T sum(p*T)/sum(p)
        """

        gm=self._get_gmix_data()

        row,col=self.get_cen()

        rowdiff=gm['row']-row
        coldiff=gm['col']-col

        p=gm['p']
        ipsum=1.0/p.sum()

        irr= ((gm['irr'] + rowdiff**2)      * p).sum()*ipsum
        icc= ((gm['icc'] + coldiff**2)      * p).sum()*ipsum

        T = irr + icc

        return T


    def get_sigma(self):
        """
        get sigma=sqrt(T/2)
        """
        T=self.get_T()
        return sqrt(T/2.)

    def get_e1e2T(self):
        """
        Get e1,e2 and T for the total gaussian mixture.
        """

        gm=self._get_gmix_data()

        row,col=self.get_cen()

        rowdiff=gm['row']-row
        coldiff=gm['col']-col

        p=gm['p']
        ipsum=1.0/p.sum()

        irr= ((gm['irr'] + rowdiff**2)      * p).sum()*ipsum
        irc= ((gm['irc'] + rowdiff*coldiff) * p).sum()*ipsum
        icc= ((gm['icc'] + coldiff**2)      * p).sum()*ipsum

        T = irr + icc

        e1=(icc-irr)/T
        e2=2.0*irc/T
        return e1,e2,T

    def get_g1g2T(self):
        """
        Get g1,g2 and T for the total gaussian mixture.
        """
        e1,e2,T=self.get_e1e2T()
        g1,g2=e1e2_to_g1g2(e1,e2)
        return g1,g2,T

    def get_e1e2sigma(self):
        """
        Get e1,e2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """

        e1,e2,T=self.get_e1e2T()
        sigma=sqrt(T/2)
        return e1,e2,sigma

    def get_g1g2sigma(self):
        """
        Get g1,g2 and sigma for the total gmix.

        Warning: only really works if the centers are the same
        """
        e1,e2,T=self.get_e1e2T()
        g1,g2=e1e2_to_g1g2(e1,e2)

        sigma=sqrt(T/2)
        return g1,g2,sigma

    def get_flux(self):
        """
        get sum(p)
        """
        gm=self._get_gmix_data()
        return gm['p'].sum()
    # alias
    get_psum=get_flux

    def set_flux(self, psum):
        """
        set a new value for sum(p)
        """
        gm=self._get_gmix_data()

        psum0 = gm['p'].sum()
        rat = psum/psum0
        gm['p'] *= rat

        # we will need to reset the pnorm values
        gm['norm_set']=0

    # alias
    set_psum=set_flux


    def set_norms(self):
        """
        Needed to actually evaluate the gaussian.  This is done internally
        by the c code so if all goes well you don't need to call this
        """
        gm=self._get_gmix_data()
        _gmix.set_norms(gm)

    def fill(self, pars):
        """
        fill the gaussian mixture from a 'full' parameter array.

        The length must match the internal size

        parameters
        ----------
        pars: array-like
            [p1,row1,col1,irr1,irc1,icc1,
             p2,row2,col2,irr2,irc2,icc2,
             ...]

             Should have length 6*ngauss
        """

        gm=self._get_gmix_data()
        pars=array(pars, dtype='f8', copy=False) 
        _gmix.gmix_fill(gm, pars, self._model)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMix(ngauss=self._ngauss)
        gmix._data[:] = self._data[:]
        return gmix

    def get_sheared(self, s1, s2=None):
        """
        Get a sheared version of the gaussian mixture

        call as either 
            gmnew = gm.get_sheared(shape)
        or
            gmnew = gm.get_sheared(g1,g2)
        """
        if isinstance(s1, Shape):
            shear1=s1.g1
            shear2=s1.g2
        elif s2 is not None:
            shear1=s1
            shear2=s2
        else:
            raise RuntimeError("send a Shape or s1,s2")

        new_gmix = self.copy()


        ndata = new_gmix._get_gmix_data()
        ndata['norm_set']=0

        for i in xrange(len(self)):
            irr=ndata['irr'][i]
            irc=ndata['irc'][i]
            icc=ndata['icc'][i]

            irr_s,irc_s,icc_s=moments.get_sheared_moments(
                irr,irc,icc,
                shear1,shear2
            )

            det = irr_s*icc_s - irc_s*irc_s
            ndata['irr'][i] = irr_s
            ndata['irc'][i] = irc_s
            ndata['icc'][i] = icc_s
            ndata['det'][i] = det

        return new_gmix


    def convolve(self, psf):
        """
        Get a new GMix that is the convolution of the GMix with the input psf

        parameters
        ----------
        psf: GMix object
        """
        if not isinstance(psf, GMix):
            raise TypeError("Can only convolve with another GMix "
                            " got type %s" % type(psf))

        ng=len(self)*len(psf)
        output = GMix(ngauss=ng)

        gm=self._get_gmix_data()
        _gmix.convolve_fill(output._data, gm, psf._data)
        return output

    def make_image(self, dims, nsub=1, npoints=None, jacobian=None, fast_exp=False):
        """
        Render the mixture into a new image

        parameters
        ----------
        dims: 2-element sequence
            dimensions [nrows, ncols]
        nsub: integer, optional
            Defines a grid for sub-pixel integration
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        dims=numpy.array(dims, ndmin=1, dtype='i8')
        if dims.size != 2:
            raise ValueError("images must have two dimensions, "
                             "got %s" % str(dims))

        image=numpy.zeros(dims, dtype='f8')
        self._fill_image(image, nsub=nsub, npoints=npoints, jacobian=jacobian, fast_exp=fast_exp)
        return image

    def make_round(self, preserve_size=False):
        """
        make a round version of the mixture

        The transformation is performed as if a shear were applied,
        so

            Tround = T * (1-g^2) / (1+g^2)

        The center of all gaussians is set to the common mean

        returns
        -------
        New round gmix
        """
        #raise RuntimeError("fix round")
        from . import shape

        gm = self.copy()


        if preserve_size:
            # make sure the psf is isotropically at least as big as the largest
            # extent

            e1,e2,T = gm.get_e1e2T()

            irr, irc, icc = moments.e2mom(e1,e2,T)

            mat=numpy.zeros( (2,2) )
            mat[0,0]=irr
            mat[0,1]=irc
            mat[1,0]=irc
            mat[1,1]=icc

            eigs=numpy.linalg.eigvals(mat)

            factor = eigs.max()/(T/2.)

        else:
            g1,g2,T=gm.get_g1g2T()
            factor = shape.get_round_factor(g1,g2)

        gdata=gm._get_gmix_data()

        # make sure the determinant gets reset
        gdata['norm_set']=0

        ngauss=len(gm)
        for i in xrange(ngauss):
            Ti = gdata['irr'][i] + gdata['icc'][i]
            gdata['irc'][i] = 0.0
            gdata['irr'][i] = 0.5*Ti*factor
            gdata['icc'][i] = 0.5*Ti*factor


        return gm


    def _fill_image(self, image, npoints=None, nsub=1, jacobian=None, fast_exp=False):
        """
        Internal routine.  Render the mixture into a new image.  No error
        checking on the image!

        parameters
        ----------
        image: 2-d double array
            image to render into
        nsub: integer, optional
            Defines a grid for sub-pixel integration
        fast_exp: bool, optional
            use fast, approximate exp function
        """

        if fast_exp:
            fexp = 1
        else:
            fexp = 0

        gm=self._get_gmix_data()
        if jacobian is not None:
            assert isinstance(jacobian,Jacobian)
            if npoints is not None:
                _gmix.render_jacob_gauleg(gm,
                                          image,
                                          npoints,
                                          jacobian._data,
                                          fexp)
            else:
                _gmix.render_jacob(gm,
                                   image,
                                   nsub,
                                   jacobian._data,
                                   fexp)
        else:
            if npoints is not None:
                _gmix.render_gauleg(gm, image, npoints, fexp)
            else:
                _gmix.render(gm, image, nsub, fexp)


    def fill_fdiff(self, obs, fdiff, start=0, nsub=1, npoints=None, nocheck=False):
        """
        Fill fdiff=(model-data)/err given the input Observation

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        fdiff: 1-d array
            The fdiff to fill
        start: int, optional
            Where to start in the array, default 0
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        if not nocheck:
            fdiff = numpy.ascontiguousarray(fdiff, dtype='f8')

        nuse=fdiff.size-start

        image=obs.image
        if nuse < image.size:
            raise ValueError("fdiff from start must have "
                             "len >= %d, got %d" % (image.size,nuse))
        assert nsub >= 1,"nsub must be >= 1"

        gm=self._get_gmix_data()
        if npoints is not None:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff_gauleg(gm,
                                                             image,
                                                             obs.weight,
                                                             obs.jacobian._data,
                                                             fdiff,
                                                             start,
                                                             npoints)
        elif nsub > 1:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff_sub(gm,
                                                          image,
                                                          obs.weight,
                                                          obs.jacobian._data,
                                                          fdiff,
                                                          start,
                                                          nsub)
        else:
            s2n_numer,s2n_denom,npix=_gmix.fill_fdiff(gm,
                                                      image,
                                                      obs.weight,
                                                      obs.jacobian._data,
                                                      fdiff,
                                                      start)

        return {'s2n_numer':s2n_numer,
                's2n_denom':s2n_denom,
                'npix':npix}

    def __call__(self, row, col, jacobian=None):
        """
        evaluate the mixture at the specified location

        no need to send jacobian unless row,col are actually image
        coords
        """

        gm=self._get_gmix_data()

        if jacobian is not None:
            assert isinstance(jacobian,Jacobian)
            return _gmix.eval_jacob(gm, jacobian._data, row, col)
        else:
            return _gmix.eval(gm, row, col)

    def get_model_s2n_sum(self, obs):
        """
        Get the s/n sum for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)

        The s/n would be sqrt(s2n_sum).  This sum can be
        added up over multiple images

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()

        s2n_sum =_gmix.get_model_s2n_sum(gm,
                                         obs.weight,
                                         obs.jacobian._data)
        return s2n_sum

    def get_model_s2n(self, obs):
        """
        Get the s/n for the model, using only the weight
        map

            s2n_sum = sum(model_i^2 * ivar_i)
            s2n = sqrt(s2n_sum)

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """
        
        s2n_sum = self.get_model_s2n_sum(obs)
        s2n = sqrt(s2n_sum)
        return s2n


    def get_model_s2n_Tvar_sums(self, obs, altweight=None):
        """

        Get the s/n sum and weighted var(T) related sums for the model, using
        only the weight map

            s2n_sum = sum(model_i^2 * ivar_i)
            r2sum = sum(model_i^2 * ivar_i * r^2 )
            r4sum = sum(model_i^2 * ivar_i * r^4 )

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()

        if altweight is not None:
            if isinstance(altweight, GMix):
                print("using altweight")
                wdata=altweight._get_gmix_data()
                res =_gmix.get_model_s2n_Tvar_sums_altweight(gm,
                                                             wdata,
                                                             obs.weight,
                                                             obs.jacobian._data)
            else:
                raise ValueError("altweight must be a GMix")

        else:

            res =_gmix.get_model_s2n_Tvar_sums(gm,
                                               obs.weight,
                                               obs.jacobian._data)

        return res

    def get_model_s2n_Tvar(self, obs, altweight=None):
        """

        Get the s/n for the model, and weighted error on T using only the
        weight map

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

        returns
        -------
        s2n, r2_mean, Tvar

        """

        s2n_sum, r2sum, r4sum = \
            self.get_model_s2n_Tvar_sums(obs, altweight=altweight)
        s2n = sqrt(s2n_sum)

        # weighted means
        r2_mean = r2sum/s2n_sum
        r4_mean = r4sum/s2n_sum

        # assume gaussian: T = 2<r^2>
        # var(T) = T^4 / nu^2 ( <r^4> )

        T = 2*r2_mean
        Tvar = T**4/( s2n**2 * r4_mean)

        #T=self.get_T()
        #Tvar = T**4 / ( s2n**2 * ( T**2 - 2*T*r2_mean + r4_mean ) )

        return s2n, r2_mean, r4_mean, Tvar


    def get_weighted_moments(self, obs, rmax=1.e20):
        """
        Get the raw weighted moments of the image, using the input
        gaussian mixture as the weight function.  The moments are *not*
        normalized

        The weight map in the observation must be accurate for accurate
        error estimates

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set

            These are moments, so there cannot be masked portions of the image,
            and the weight map of the observation is ignored.

        returns
        --------

        In the following, W is the weight function, I is the image

           Returns the folling in the 'pars' field, in this order
               sum(W * I * F[i])
           where
               F = {
                  v,
                  u,
                  u^2-v^2,
                  2*v*u,
                  u^2+v^2,
                  1.0
               }

        where v,u are in sky coordinates relative to the jacobian center.

        Also returned are the covariance sums in a 6x6 matrix

            sum( W^2 * V * F[i]*F[j] )

        where V is the variance from the weight map
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        pars=zeros(6)
        pcov=zeros( (6,6) )
        flags,wsum,s2n_numer,s2n_denom=_gmix.get_weighted_moments(
            obs.image,
            obs.weight,
            obs.jacobian._data,
            gm,

            pars, # these get modified internally
            pcov,
            rmax,
        )

        flagstr=_moms_flagmap[flags]
        return {
            'flags':flags,
            'flagstr':flagstr,

            'pars':pars,
            'pars_cov':pcov,

            'wsum':wsum,
            'npix':obs.image.size,

            's2n_numer_sum':s2n_numer,
            's2n_denom_sum':s2n_denom,
        }

    def get_weighted_gmix_moments(self, gm, dims, jacobian=None):
        """

        Get the weighted moments of this gmix against another.  The moments are
        *not* normalized

        This parallels the get_weighted_moments method

        parameters
        ----------
        gm: GMix
            The Gaussian mixture for which to measure moments
        dims: int
            [nrow,ncol] Number of image rows and columns to mimic for evaluation
        jacobian: Jacobian, optional
            Transformation between pixel and sky coords.  The gaussian mixture
            centers should be set relative to the jacobian center.  If not
            sent, UnitJacobian(row=0.,col=0.) is used.

        returns
        --------

        In the following, W is the weight function, I is the image

           Returns the folling in the 'pars' field, in this order
               sum(W * I * F[i])
           where
               F = {
                  v,
                  u,
                  u^2-v^2,
                  2*v*u,
                  u^2+v^2,
                  1.0
               }

        where v,u are in sky coordinates relative to the jacobian center.
        """

        assert isinstance(gm,GMix)
        if jacobian is None:
            jacobian=UnitJacobian(row=0.0, col=0.0)
        else:
            assert isinstance(jacobian,Jacobian)

        # use self as the weight
        wt_gmdata=self._get_gmix_data()
        gmdata=gm._get_gmix_data()

        pars=zeros(6)
        wsum=_gmix.get_weighted_gmix_moments(
            gmdata,
            wt_gmdata,
            jacobian._data,
            dims[0],
            dims[1],
            pars, # these get modified internally
        )

        flags=0
        flagstr=_moms_flagmap[flags]
        return {
            'flags':flags,
            'flagstr':flagstr,

            'wsum':wsum,
            'pars':pars,
        }


    def get_loglike(self, obs, nsub=1, npoints=None, more=False):
        """
        Calculate the log likelihood given the input Observation


        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        nsub: int, optional
            Integrate the model over each pixel using a nsubxnsub grid
        more:
            if True, return a dict with more informatioin
        """

        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        if npoints is not None:
            loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_gauleg(gm,
                                                                      obs.image,
                                                                      obs.weight,
                                                                      obs.jacobian._data,
                                                                      npoints)

        elif nsub > 1:
            #print("doing nsub")
            loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_sub(gm,
                                                                   obs.image,
                                                                   obs.weight,
                                                                   obs.jacobian._data,
                                                                   nsub)

        else:
            if obs.has_aperture():
                aperture=obs.get_aperture()
                #print("using aper:",aperture)
                loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_aper(gm,
                                                                        obs.image,
                                                                        obs.weight,
                                                                        obs.jacobian._data,
                                                                        aperture)


            else:
                loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike(gm,
                                                                   obs.image,
                                                                   obs.weight,
                                                                   obs.jacobian._data)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def get_loglike_robust(self, obs, nu, nsub=1, more=False):
        """
        Calculate the log likelihood given the input Observation
        using robust likelihood

        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
        nu: parameter for robust likelihood - nu > 2, nu -> \infty is a Gaussian (or chi^2)
        """
        #print("using robust")
        assert nsub==1,"nsub must be 1 for robust"


        if obs.jacobian is not None:
            assert isinstance(obs.jacobian,Jacobian)

        gm=self._get_gmix_data()
        loglike,s2n_numer,s2n_denom,npix=_gmix.get_loglike_robust(gm,
                                                                  obs.image,
                                                                  obs.weight,
                                                                  obs.jacobian._data,
                                                                  nu)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def get_loglike_margsky(self, obs, model_image, nsub=1, more=False):
        """
        Calculate the log likelihood given the input Observation, subtracting
        the mean of the image and model.  The model is first rendered into the
        input image so that rendering does not happen twice


        parameters
        ----------
        obs: Observation
            The Observation to compare with. See ngmix.observation.Observation
            The Observation must have a weight map set
            The Observation must have image_mean set
        model_image: 2-d double array
            image to render model into
        nsub: integer, optional
            Defines a grid for sub-pixel integration 
        """

        #print("using margsky")
        image=obs.image

        dt=model_image.dtype.descr[0][1]

        mess="image must be '%s', got '%s'"
        assert dt == self._f8_type,mess % (self._f8_type,dt)

        assert len(model_image.shape)==2,"image must be 2-d"
        assert model_image.shape==image.shape,"image and model must be same shape"

        model_image[:,:]=0
        self._fill_image(model_image, nsub=nsub, jacobian=obs.jacobian)

        model_mean=_gmix.get_image_mean(model_image, obs.weight)

        loglike,s2n_numer,s2n_denom,npix=\
                _gmix.get_loglike_images_margsky(image,
                                                 obs.image_mean,
                                                 obs.weight,
                                                 model_image,
                                                 model_mean)

        if more:
            return {'loglike':loglike,
                    's2n_numer':s2n_numer,
                    's2n_denom':s2n_denom,
                    'npix':npix}
        else:
            return loglike

    def _get_gmix_data(self):
        """
        same as get_data for normal models, but not all
        """
        return self._data

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._data = zeros(self._ngauss, dtype=_gauss2d_dtype)

    def make_galsim_object(self):
        """
        make a galsim representation for the gaussian mixture
        """
        import galsim

        data = self._get_gmix_data()

        row,col = self.get_cen()

        gsobjects=[]
        for i in xrange(len(self)):
            flux = data['p'][i]
            T = data['irr'][i] + data['icc'][i]
            e1 = (data['icc'][i] - data['irr'][i])/T
            e2 = 2.0*data['irc'][i]/T

            rowshift = data['row'][i]-row
            colshift = data['col'][i]-col

            g1,g2=e1e2_to_g1g2(e1,e2)

            Tround = moments.get_Tround(T, g1, g2)
            sigma_round = sqrt(Tround/2.0)

            gsobj = galsim.Gaussian(flux=flux, sigma=sigma_round)

            gsobj = gsobj.shear(g1=g1, g2=g2)
            gsobj = gsobj.shift(colshift, rowshift)

            gsobjects.append( gsobj )

        gs_obj = galsim.Add(gsobjects)

        #rowshift = row-int(row)-0.5
        #colshift = col-int(col)-0.5
        #gs_obj = gs_obj.shift(colshift, rowshift)

        return gs_obj

    def __len__(self):
        return self._ngauss

    def __repr__(self):
        rep=[]
        #fmt="p: %-10.5g row: %-10.5g col: %-10.5g irr: %-10.5g irc: %-10.5g icc: %-10.5g"
        fmt="p: %.4g row: %.4g col: %.4g irr: %.4g irc: %.4g icc: %.4g"
        for i in xrange(self._ngauss):
            t=self._data[i]
            s=fmt % (t['p'],t['row'],t['col'],t['irr'],t['irc'],t['icc'])
            rep.append(s)

        rep='\n'.join(rep)
        return rep

class GMixList(list):
    """
    Hold a list of GMix objects

    This class provides a bit of type safety and ease of type checking
    """
    def append(self, gmix):
        """
        Add a new mixture

        over-riding this for type safety
        """
        assert isinstance(gmix,GMix),"gmix should be of type GMix"
        super(GMixList,self).append(gmix)

    def __setitem__(self, index, gmix):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix,GMix),"gmix should be of type GMix"
        super(GMixList,self).__setitem__(index, gmix)

class MultiBandGMixList(list):
    """
    Hold a list of lists of GMixList objects, each representing a filter
    band

    This class provides a bit of type safety and ease of type checking
    """

    def append(self, gmix_list):
        """
        add a new GMixList

        over-riding this for type safety
        """
        assert isinstance(gmix_list,GMixList),"gmix_list should be of type GMixList"
        super(MultiBandGMixList,self).append(gmix_list)

    def __setitem__(self, index, gmix_list):
        """
        over-riding this for type safety
        """
        assert isinstance(gmix_list,GMixList),"gmix_list should be of type GMixList"
        super(MultiBandGMixList,self).__setitem__(index, gmix_list)



class GMixModel(GMix):
    """
    A two-dimensional gaussian mixture created from a set of model parameters

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, pars, model):

        self._set_f8_type()
        self._model      = _gmix_model_dict[model]
        self._model_name = _gmix_string_dict[self._model]

        self._ngauss = _gmix_ngauss_dict[self._model]
        self._npars  = _gmix_npars_dict[self._model]

        self.reset()
        self.fill(pars)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixModel(self._pars, self._model_name)
        return gmix

    def set_cen(self, row, col):
        """
        Move the mixture to a new center

        set pars as well
        """
        gm=self._get_gmix_data()

        pars=self._pars
        row0,col0=self.get_cen()

        row_shift = row - row0
        col_shift = col - col0

        gm['row'] += row_shift
        gm['col'] += col_shift

        pars[0] = row
        pars[1] = col

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """


        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != self._npars:
            err="model '%s' requires %s pars, got %s"
            err =err % (self._model_name,self._npars, pars.size)
            raise GMixFatalError(err)

        self._pars = pars

        gm=self._get_gmix_data()
        _gmix.gmix_fill(gm, pars, self._model)


_sersic_data_10gauss = array([ ( 0.55      , [  8.01401565e-08,   1.61178975e-06,   3.47387023e-06,   1.05255961e-05,   9.70462516e-05,   7.92169712e-04,   6.47729686e-03,   5.62902638e-02,   4.05121325e-01,   5.31206207e-01], [  4.21562137e-04,   2.38531597e-03,   8.32182810e-03,   8.64966265e-03,   2.40810967e-02,   6.78571708e-02,   1.84287752e-01,   4.43207499e-01,   8.45220230e-01,   1.18858756e+00]),
       ( 0.59723947, [  2.86071710e-07,   5.33055124e-06,   5.71337083e-05,   4.57141101e-04,   3.05143566e-03,   1.94627006e-02,   1.21151401e-01,   2.78238807e-03,   4.88705430e-01,   3.64326753e-01], [  3.97876982e-04,   2.37809257e-03,   9.44698214e-03,   3.06001758e-02,   8.55310271e-02,   2.13360023e-01,   4.77224129e-01,   5.66338401e-01,   8.98796543e-01,   1.36397617e+00]),
       ( 0.64853634, [  1.03368868e-07,   2.03008970e-06,   2.35983216e-05,   1.94945537e-04,   1.26713111e-03,   7.41453789e-03,   4.07256458e-02,   1.93710970e-01,   4.98421956e-01,   2.58239081e-01], [  1.24584295e-04,   8.06673411e-04,   3.52145224e-03,   1.24198748e-02,   3.69141741e-02,   9.78964885e-02,   2.36672800e-01,   5.15914716e-01,   9.72540327e-01,   1.56797497e+00]),
       ( 0.70423909, [  1.04024254e-05,   1.54755195e-04,   1.28923928e-03,   8.11094440e-03,   4.22668818e-02,   9.00662232e-04,   1.75563372e-01,   4.32308637e-01,   3.25469368e-01,   1.39257376e-02], [  1.05364185e-03,   6.41364745e-03,   2.43928412e-02,   7.38445371e-02,   1.91625146e-01,   3.19384229e-01,   4.37847082e-01,   8.72876608e-01,   1.53411651e+00,   2.68938051e+00]),
       ( 0.76472615, [  5.83711595e-06,   6.48417969e-05,   4.55396064e-05,   7.76945171e-04,   4.66924941e-03,   2.31353771e-02,   9.57447998e-02,   2.89434765e-01,   4.35174344e-01,   1.50948301e-01], [  4.73775032e-04,   2.85467452e-03,   4.93180847e-03,   1.31832304e-02,   4.06812187e-02,   1.09503436e-01,   2.64027139e-01,   5.76389701e-01,   1.13733404e+00,   2.05514704e+00]),
       ( 0.83040843, [  1.08341570e-05,   1.52458342e-04,   1.18400095e-03,   6.77890259e-03,   3.14315108e-02,   1.17386492e-01,   3.08588212e-01,   3.25891386e-04,   4.04191043e-01,   1.29950654e-01], [  4.40845748e-04,   2.94760833e-03,   1.19885290e-02,   3.83752152e-02,   1.05589902e-01,   2.59971169e-01,   5.83028888e-01,   9.56857940e-01,   1.20023355e+00,   2.31270811e+00]),
       ( 0.90173217, [  1.58304350e-05,   2.17368521e-04,   1.64320501e-03,   9.05174721e-03,   3.95809031e-02,   1.35258957e-01,   3.36489841e-04,   3.18616387e-01,   3.78306733e-01,   1.16972379e-01], [  3.62418502e-04,   2.53551087e-03,   1.06726930e-02,   3.51533845e-02,   9.93142416e-02,   2.51213182e-01,   3.55511466e-01,   5.82362987e-01,   1.25394224e+00,   2.57931931e+00]),
       ( 0.9791819 , [  8.05926951e-06,   1.07059644e-04,   7.86741526e-04,   4.22099045e-03,   1.81495308e-02,   6.37714152e-02,   1.75443263e-01,   3.30738224e-01,   3.20677411e-01,   8.60973051e-02], [  1.50395352e-04,   1.09537569e-03,   4.75726197e-03,   1.61252335e-02,   4.68593568e-02,   1.22228602e-01,   2.94019154e-01,   6.64138205e-01,   1.43094932e+00,   3.03343076e+00]),
       ( 1.06328378, [  1.04683755e-05,   1.35401052e-04,   9.69525360e-04,   5.04695445e-03,   2.08917566e-02,   6.99794775e-02,   1.81958068e-01,   3.25443458e-01,   3.08889237e-01,   8.66756527e-02], [  1.14802889e-04,   8.71717787e-04,   3.90935934e-03,   1.36363934e-02,   4.07494596e-02,   1.09475448e-01,   2.72213624e-01,   6.39714184e-01,   1.45029143e+00,   3.29634935e+00]),
       ( 1.15460917, [  1.31634384e-05,   1.66160461e-04,   1.16186881e-03,   5.87761845e-03,   2.34631595e-02,   7.51470791e-02,   1.85955468e-01,   3.19090303e-01,   3.00088728e-01,   8.90364513e-02], [  8.60851176e-05,   6.81957892e-04,   3.16335420e-03,   1.13745226e-02,   3.50037218e-02,   9.69301882e-02,   2.49182584e-01,   6.09170730e-01,   1.45258453e+00,   3.54015070e+00]),
       ( 1.25377849, [  1.60985445e-05,   1.98254638e-04,   1.35142676e-03,   6.63780914e-03,   2.55788097e-02,   7.87039207e-02,   1.87113058e-01,   3.11986206e-01,   2.94636912e-01,   9.37775043e-02], [  6.34510584e-05,   5.24146242e-04,   2.51322137e-03,   9.31123625e-03,   2.95041100e-02,   8.42324124e-02,   2.24040731e-01,   5.70281862e-01,   1.43112430e+00,   3.74340872e+00]),
       ( 1.36146546, [  1.89093426e-05,   2.27263320e-04,   1.51311955e-03,   7.24165320e-03,   2.70920665e-02,   8.06921574e-02,   1.85995800e-01,   3.04455556e-01,   2.91856693e-01,   1.00906782e-01], [  4.54315864e-05,   3.91325790e-04,   1.94128065e-03,   7.42413721e-03,   2.42819511e-02,   7.16572464e-02,   1.97606530e-01,   5.24361651e-01,   1.38542708e+00,   3.89228759e+00]),
       ( 1.47840165, [  2.15943168e-05,   2.54269785e-04,   1.65743637e-03,   7.74187904e-03,   2.81838213e-02,   8.15858222e-02,   1.83394560e-01,   2.96689795e-01,   2.90539775e-01,   1.09931048e-01], [  3.18037898e-05,   2.86285310e-04,   1.47244124e-03,   5.82089594e-03,   1.96687312e-02,   6.00374759e-02,   1.71742263e-01,   4.75152625e-01,   1.32117203e+00,   3.98592872e+00]),
       ( 1.60538149, [  2.40724173e-05,   2.77714171e-04,   1.77311063e-03,   8.09874210e-03,   2.87864564e-02,   8.13806571e-02,   1.79484632e-01,   2.88824589e-01,   2.90480675e-01,   1.20869352e-01], [  2.18125378e-05,   2.05249676e-04,   1.09478774e-03,   4.47715775e-03,   1.56464440e-02,   4.94580983e-02,   1.46910049e-01,   4.24071906e-01,   1.24111996e+00,   4.02181813e+00]),
       ( 1.74326762, [  2.61941960e-05,   2.96623881e-04,   1.85823585e-03,   8.31870122e-03,   2.89604866e-02,   8.02968331e-02,   1.74646360e-01,   2.80957060e-01,   2.91177047e-01,   1.33462458e-01], [  1.46401913e-05,   1.44231944e-04,   7.98894745e-04,   3.38413787e-03,   1.22466244e-02,   4.01331020e-02,   1.23913608e-01,   3.73490349e-01,   1.15088911e+00,   4.00640596e+00]),
       ( 1.89299678, [  2.78754745e-05,   3.10624586e-04,   1.91349678e-03,   8.41454016e-03,   2.87787600e-02,   7.85579329e-02,   1.69220633e-01,   2.73178275e-01,   2.92225556e-01,   1.47372306e-01], [  9.62139433e-06,   9.94697850e-05,   5.73162501e-04,   2.51853858e-03,   9.44994295e-03,   3.21455477e-02,   1.03285515e-01,   3.25371448e-01,   1.05616510e+00,   3.95039994e+00]),
       ( 2.05558618, [  2.90350248e-05,   3.19276117e-04,   1.93820203e-03,   8.39479635e-03,   2.82940696e-02,   7.63058232e-02,   1.63388322e-01,   2.65511931e-01,   2.93435791e-01,   1.62382754e-01], [  6.18998042e-06,   6.73512627e-05,   4.04484307e-04,   1.84680737e-03,   7.19631818e-03,   2.54465539e-02,   8.51925700e-02,   2.80805571e-01,   9.60973110e-01,   3.86357379e+00]),
       ( 2.23214036, [  2.96140088e-05,   3.22437789e-04,   1.93368763e-03,   8.27042549e-03,   2.75512202e-02,   7.36518287e-02,   1.57289649e-01,   2.57984191e-01,   2.94689108e-01,   1.78277838e-01], [  3.89534035e-06,   4.47801464e-05,   2.80927115e-04,   1.33523641e-03,   5.41213784e-03,   1.99243776e-02,   6.96065595e-02,   2.40393161e-01,   8.68359605e-01,   3.75542853e+00]),
       ( 2.42385878, [  2.95598234e-05,   3.19984720e-04,   1.90116807e-03,   8.05159601e-03,   2.65880864e-02,   7.06737343e-02,   1.50994665e-01,   2.50578489e-01,   2.95943945e-01,   1.94918772e-01], [  2.39372763e-06,   2.92129002e-05,   1.91967997e-04,   9.51840191e-04,   4.02061896e-03,   1.54360219e-02,   5.63629646e-02,   2.04283566e-01,   7.80133182e-01,   3.63340518e+00]),
       ( 2.63204388, [  2.89473876e-05,   3.12549299e-04,   1.84424273e-03,   7.75313779e-03,   2.54470153e-02,   6.74470255e-02,   1.44560852e-01,   2.43261744e-01,   2.97158874e-01,   2.12185612e-01], [  1.43859294e-06,   1.87157587e-05,   1.29141629e-04,   6.69314604e-04,   2.95154168e-03,   1.18371924e-02,   4.52505203e-02,   1.72419149e-01,   6.97411214e-01,   3.50351138e+00]),
       ( 2.85810999, [  2.82753633e-05,   3.03313581e-04,   1.77639146e-03,   7.41714750e-03,   2.42260621e-02,   6.41307219e-02,   1.38128102e-01,   2.36010562e-01,   2.98165464e-01,   2.29813959e-01], [  8.60498713e-07,   1.18933983e-05,   8.61136439e-05,   4.66607847e-04,   2.14940253e-03,   9.01226478e-03,   3.61050636e-02,   1.44809448e-01,   6.21373295e-01,   3.37199102e+00]),
       ( 3.10359291, [  2.93236948e-05,   3.02780513e-04,   1.73866833e-03,   7.16260190e-03,   2.31889825e-02,   6.11569793e-02,   1.32146280e-01,   2.28914923e-01,   2.98429095e-01,   2.46930365e-01], [  5.48977034e-07,   7.80787491e-06,   5.85389112e-05,   3.29297070e-04,   1.57776146e-03,   6.89704875e-03,   2.89036345e-02,   1.21881970e-01,   5.54498606e-01,   3.24925823e+00]),
       ( 3.37016034, [  3.48953971e-05,   3.25267689e-04,   1.78311035e-03,   7.13349007e-03,   2.26479917e-02,   5.90446094e-02,   1.27207672e-01,   2.22225294e-01,   2.97365119e-01,   2.62232551e-01], [  4.07185896e-07,   5.60177132e-06,   4.22209090e-05,   2.42446010e-04,   1.19494543e-03,   5.40299467e-03,   2.35432745e-02,   1.03881523e-01,   4.99111815e-01,   3.14664857e+00]),
       ( 3.65962323, [  4.59041652e-05,   3.70330746e-04,   1.89971749e-03,   7.28886987e-03,   2.24987331e-02,   5.76139749e-02,   1.23139123e-01,   2.15987416e-01,   2.95335990e-01,   2.75819941e-01], [  3.38268095e-07,   4.33415029e-06,   3.20884019e-05,   1.85446881e-04,   9.31173239e-04,   4.32415684e-03,   1.94849500e-02,   8.95660014e-02,   4.52807409e-01,   3.06088707e+00]),
       ( 3.97394807, [  6.33540691e-05,   4.35727119e-04,   2.07038132e-03,   7.56486990e-03,   2.25873168e-02,   5.65988196e-02,   1.19646879e-01,   2.10143076e-01,   2.92763057e-01,   2.88126519e-01], [  2.98370104e-07,   3.52047152e-06,   2.52780042e-05,   1.45697831e-04,   7.40403338e-04,   3.51351948e-03,   1.63076936e-02,   7.78448364e-02,   4.13042091e-01,   2.98670903e+00]),
       ( 4.31527026, [  8.95660906e-05,   5.23587419e-04,   2.29231307e-03,   7.94133092e-03,   2.28574042e-02,   5.58952159e-02,   1.16609377e-01,   2.04661407e-01,   2.89822687e-01,   2.99307111e-01], [  2.72982658e-07,   2.96635655e-06,   2.05033258e-05,   1.17089276e-04,   5.99045576e-04,   2.89278185e-03,   1.37828302e-02,   6.81319057e-02,   3.78540272e-01,   2.92195830e+00]),
       ( 4.6859086 , [  1.28194060e-04,   6.38545324e-04,   2.56951043e-03,   8.41572146e-03,   2.32907767e-02,   5.54622712e-02,   1.13973058e-01,   1.99525631e-01,   2.86594641e-01,   3.09401651e-01], [  2.56398945e-07,   2.57816748e-06,   1.70788285e-05,   9.61403333e-05,   4.92956499e-04,   2.41309566e-03,   1.17638822e-02,   6.00516397e-02,   3.48539274e-01,   2.86566581e+00]),
       ( 5.08838104, [  1.84471432e-04,   7.86978019e-04,   2.90897950e-03,   8.99176417e-03,   2.38817104e-02,   5.52793328e-02,   1.11706359e-01,   1.94723202e-01,   2.83123794e-01,   3.18413408e-01], [  2.45927290e-07,   2.30344703e-06,   1.45902185e-05,   8.06321782e-05,   4.12699023e-04,   2.04048083e-03,   1.01451431e-02,   5.33264557e-02,   3.22489668e-01,   2.81726486e+00]),
       ( 5.52542181, [  2.65622826e-04,   9.76822339e-04,   3.31941116e-03,   9.67555735e-03,   2.46286425e-02,   5.53324240e-02,   1.09784707e-01,   1.90241837e-01,   2.79439673e-01,   3.26335303e-01], [  2.40147758e-07,   2.11026443e-06,   1.27705987e-05,   6.90727444e-05,   3.51665300e-04,   1.75016026e-03,   8.84632015e-03,   4.77375914e-02,   2.99955552e-01,   2.77635125e+00]),
       ( 6.        , [  3.81365929e-04,   1.21752907e-03,   3.81056295e-03,   1.04738397e-02,   2.55304874e-02,   5.56082090e-02,   1.08184364e-01,   1.86068398e-01,   2.75567761e-01,   3.33157483e-01], [  2.38270293e-07,   1.97827368e-06,   1.14419190e-05,   6.04297403e-05,   3.05109443e-04,   1.52360257e-03,   7.80464927e-03,   4.31058930e-02,   2.80569457e-01,   2.74262465e+00])],
      dtype=[('n', '<f8'), ('p', '<f8', (10,)), ('f', '<f8', (10,))])


class GMixSersicBase(GMixModel):
    def __init__(self, pars, model):
        """
        [c1,c2, g1,g2, T, n, flux]
        """
        assert len(pars)==7
        ngauss=_gmix_ngauss_dict[model]

        self._send_pars = zeros(6)
        self._pvals = zeros(ngauss)
        self._fvals = zeros(ngauss)

        super(GMixSersicBase,self).__init__(pars, model)

    def plot_spline(self):
        """
        plot the spline fits
        """
        import biggles
        import pcolors

        pkey=biggles.PlotKey(0.9, 0.9, halign='right')
        pplt=biggles.FramedPlot(
            key=pkey,
            ylabel='pval',
            xlabel='n',
        )
        size=0.75
        colors=pcolors.rainbow(args.ngauss)
        nn=numpy.linspace(data['n'].min(), data['n'].max(), 1000)
        ns=data['n'].argsort()
        for igauss in range(args.ngauss):

            points = biggles.Points(
                data['n'],
                data['pvals'][:,igauss],
                color=colors[igauss],
                type='filled circle',
                size=size,
                label='gauss %d' % igauss,
            )
            pplt.add(points)

            spline=scipy.interpolate.CubicSpline(
                data['n'][ns],
                data['pvals'][ns,igauss],
            )
            curve = biggles.Curve(
                nn,
                spline(nn),
                color=colors[igauss],
            )
            pplt.add(curve)



        fkey=biggles.PlotKey(0.9, 0.9, halign='right')
        fplt=biggles.FramedPlot(
            key=fkey,
            ylabel='fval',
            xlabel='n',
            ylog=True,
        )
        for igauss in range(args.ngauss):
            points = biggles.Points(
                data['n'],
                data['fvals'][:,igauss].clip(min=1.0e-5),
                color=colors[igauss],
                type='filled circle',
                size=size,
                label='gauss %d' % igauss,
            )
            fplt.add(points)

            spline=scipy.interpolate.CubicSpline(
                data['n'][ns],
                data['fvals'][ns,igauss],
            )
            curve = biggles.Curve(
                nn,
                spline(nn),
                color=colors[igauss],
            )
            fplt.add(curve)



        tab=biggles.Table(2,1)
        #tab.aspect_ratio = 0.5
        tab[0,0] = pplt
        tab[1,0] = fplt

        h=800
        tab.show(width=h, height=h)

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != self._npars:
            err="model '%s' requires %s pars, got %s"
            err =err % (self._model_name,self._npars, pars.size)
            raise GMixFatalError(err)

        self._pars = pars

        # we don't send the n value
        n = pars[5]

        fvals, pvals = self._fill_pvals_fvals(n)

        send_pars=self._send_pars
        send_pars[0:0+5] = pars[0:0+5]
        send_pars[5] = pars[6]

        gm=self._get_gmix_data()
        _gmix.gmix_fill_fvals_pvals(
            gm,
            send_pars,
            fvals,
            pvals,
            self._model,
        )



class GMixSersic5(GMixSersicBase):
    """
    approximate sersic profiles using gaussians
    """

    p0spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['p'][:,0],
    )
    p1spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['p'][:,1],
    )
    p2spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['p'][:,2],
    )
    p3spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['p'][:,3],
    )
    p4spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['p'][:,4],
    )



    f0spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['f'][:,0],
    )
    f1spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['f'][:,1],
    )
    f2spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['f'][:,2],
    )
    f3spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['f'][:,3],
    )
    f4spline = scipy.interpolate.CubicSpline(
        _sersic_data_5gauss['n'],
        _sersic_data_5gauss['f'][:,4],
    )

    def __init__(self, pars):
        """
        [c1,c2, g1,g2, T, n, flux]
        """
        super(GMixSersic5,self).__init__(pars, 'sersic5')
        assert self._ngauss==5

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixSersic5(self._pars)
        return gmix

    def _fill_pvals_fvals(self, n):
        pvals=self._pvals
        fvals=self._fvals

        pvals[0] = GMixSersic5.p0spline(n)
        pvals[1] = GMixSersic5.p1spline(n)
        pvals[2] = GMixSersic5.p2spline(n)
        pvals[3] = GMixSersic5.p3spline(n)
        pvals[4] = GMixSersic5.p4spline(n)

        fvals[0] = GMixSersic5.f0spline(n)
        fvals[1] = GMixSersic5.f1spline(n)
        fvals[2] = GMixSersic5.f2spline(n)
        fvals[3] = GMixSersic5.f3spline(n)
        fvals[4] = GMixSersic5.f4spline(n)

        return fvals, pvals

class GMixSersic10(GMixSersicBase):
    """
    approximate sersic profiles using gaussians
    """
    p0spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,0],
    )
    p1spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,1],
    )
    p2spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,2],
    )
    p3spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,3],
    )
    p4spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,4],
    )
    p5spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,5],
    )
    p6spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,6],
    )
    p7spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,7],
    )
    p8spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,8],
    )
    p9spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['p'][:,9],
    )




    f0spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,0],
    )
    f1spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,1],
    )
    f2spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,2],
    )
    f3spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,3],
    )
    f4spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,4],
    )
    f5spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,5],
    )
    f6spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,6],
    )
    f7spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,7],
    )
    f8spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,8],
    )
    f9spline = scipy.interpolate.CubicSpline(
        _sersic_data_10gauss['n'],
        _sersic_data_10gauss['f'][:,9],
    )

    def __init__(self, pars):
        """
        [c1,c2, g1,g2, T, n, flux]
        """
        super(GMixSersic10,self).__init__(pars, 'sersic10')
        assert self._ngauss==10

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixSersic10(self._pars)
        return gmix

    def _fill_pvals_fvals(self, n):
        pvals=self._pvals
        fvals=self._fvals

        pvals[0] = GMixSersic10.p0spline(n)
        pvals[1] = GMixSersic10.p1spline(n)
        pvals[2] = GMixSersic10.p2spline(n)
        pvals[3] = GMixSersic10.p3spline(n)
        pvals[4] = GMixSersic10.p4spline(n)
        pvals[5] = GMixSersic10.p5spline(n)
        pvals[6] = GMixSersic10.p6spline(n)
        pvals[7] = GMixSersic10.p7spline(n)
        pvals[8] = GMixSersic10.p8spline(n)
        pvals[9] = GMixSersic10.p9spline(n)

        fvals[0] = GMixSersic10.f0spline(n)
        fvals[1] = GMixSersic10.f1spline(n)
        fvals[2] = GMixSersic10.f2spline(n)
        fvals[3] = GMixSersic10.f3spline(n)
        fvals[4] = GMixSersic10.f4spline(n)
        fvals[5] = GMixSersic10.f5spline(n)
        fvals[6] = GMixSersic10.f6spline(n)
        fvals[7] = GMixSersic10.f7spline(n)
        fvals[8] = GMixSersic10.f8spline(n)
        fvals[9] = GMixSersic10.f9spline(n)

        return fvals, pvals

GMixSersic = GMixSersic10

class GMixCM(GMix):
    """
    Composite Model exp and dev using just fracdev

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    model: string or gmix type
        e.g. 'exp' or GMIX_EXP
    """
    def __init__(self, fracdev, TdByTe, pars):

        self._fracdev = fracdev
        self._TdByTe = TdByTe
        self._Tfactor = _gmix.get_cm_Tfactor(fracdev, TdByTe)

        self._model      = _gmix_model_dict['fracdev']
        self._model_name = _gmix_string_dict[self._model]

        self._ngauss = _gmix_ngauss_dict[self._model]
        self._npars  = _gmix_npars_dict[self._model]

        self.reset()

        self.fill(pars)


    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        #gmix = GMixCM(self._exp_pars, self._dev_pars, self._fracdev)
        gmix = GMixCM(self._fracdev,
                      self._TdByTe,
                      self._pars)
        return gmix

    def reset(self):
        """
        Replace the data array with a zeroed one.
        """
        self._data = zeros(self._ngauss, dtype=_cm_dtype)
        self._data['fracdev'][0] = self._fracdev
        self._data['TdByTe'][0] = self._TdByTe
        self._data['Tfactor'][0] = self._Tfactor


    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters

        parameters
        ----------
        pars: ndarray or sequence
            The parameters
        """

        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != 6:
            raise GMixFatalError("must have 6 pars")

        self._pars = pars

        data=self.get_data()
        _gmix.gmix_fill_cm(data, pars)

    def _get_gmix_data(self):
        """
        same as get_data for normal models, but not all
        """
        return self._data['gmix'][0]


    def __repr__(self):
        rep=[]
        #fmt="p: %-10.5g row: %-10.5g col: %-10.5g irr: %-10.5g irc: %-10.5g icc: %-10.5g"
        fmt="p: %.4g row: %.4g col: %.4g irr: %.4g irc: %.4g icc: %.4g"

        gm=self._get_gmix_data()
        for i in xrange(self._ngauss):
            t=gm[i]
            s=fmt % (t['p'],t['row'],t['col'],t['irr'],t['irc'],t['icc'])
            rep.append(s)

        rep='\n'.join(rep)
        return rep



def get_coellip_npars(ngauss):
    return 4 + 2*ngauss

def get_coellip_ngauss(npars):
    return (npars-4)//2

class GMixCoellip(GMixModel):
    """
    A two-dimensional gaussian mixture, each co-centeric and co-elliptical

    Inherits from the more general GMix class, and all its methods.

    parameters
    ----------
    pars: array-like
        Parameter array. The number of elements will depend
        on the model type.
    """


    def __init__(self, pars):

        self._model      = GMIX_COELLIP
        self._model_name = 'coellip'
        pars = array(pars, dtype='f8', copy=True) 

        npars=pars.size

        ncheck=npars-4
        if ( ncheck % 2 ) != 0:
            raise ValueError("coellip must have len(pars)==4+2*ngauss, got %s" % npars)

        self._pars=pars
        self._ngauss = ncheck//2
        self._npars = npars

        self.reset()
        gm=self._get_gmix_data()
        _gmix.gmix_fill(gm, pars, self._model)

    def fill(self, pars):
        """
        Fill in the gaussian mixture with new parameters
        """

        pars = array(pars, dtype='f8', copy=True) 

        if pars.size != self._npars:
            raise ValueError("input pars have size %d, "
                             "expected %d" % (pars.size, self._npars))

        self._pars[:]=pars[:]

        gm=self._get_gmix_data()
        _gmix.gmix_fill(self._data, pars, self._model)

    def copy(self):
        """
        Get a new GMix with the same parameters
        """
        gmix = GMixCoellip(self._pars)
        return gmix


def cbinary_search(a, x):
    """
    use weave inline for speed
    """
    import scipy.weave
    from scipy.weave import inline
    from scipy.weave.converters import blitz

    size=a.size

    code="""
    long up=size;
    long down=-1;

    for (;;) {
        if ( x < a(0) ) {
            return_val =  0;
            break;
        }
        if (x > a(up-1)) {
            return_val =  up-1;
            break;
        }

        long mid=0;
        double val=0;
        while ( (up-down) > 1 ) {
            mid = down + (up-down)/2;
            val=a(mid);

            if (x >= val) {
                down=mid;
            } else {
                up=mid;
            }
     
        }
        return_val = down;
    }
    """
    down=inline(code, ['a','x','size'],
                type_converters=blitz,
                compiler='gcc')

    return down



def cinterp_multi_scalar(xref, yref, xinterp, output):
    import scipy.weave
    from scipy.weave import inline
    from scipy.weave.converters import blitz

    npoints=xref.size
    ndim=yref.shape[1]

    ilo = cbinary_search(xref, xinterp)

    code="""
    double x=xinterp;

    if (ilo < 0) {
        ilo=0;
    }
    if (ilo >= (npoints-1)) {
        ilo=npoints-2;
    }

    int ihi = ilo+1;

    double xlo=xref(ilo);
    double xhi=xref(ihi);
    double xdiff = xhi-xlo;
    double xmxlo = x-xlo;

    for (int i=0; i<ndim; i++) {

        double ylo = yref(ilo, i);
        double yhi = yref(ihi, i);
        double ydiff = yhi - ylo;

        double slope = ydiff/xdiff;

        output(i) = xmxlo*slope + ylo;

    }

    return_val=1;
    """

    inline(code, ['xref','yref','xinterp','ilo','npoints','ndim','output'],
           type_converters=blitz)#, compiler='gcc')











GMIX_FULL=0
GMIX_GAUSS=1
GMIX_TURB=2
GMIX_EXP=3
GMIX_DEV=4
GMIX_BDC=5
GMIX_BDF=6
GMIX_COELLIP=7
GMIX_SERSIC=8
GMIX_FRACDEV=9

# Composite Model
GMIX_CM=10

# moments
GMIX_GAUSSMOM=11

GMIX_SERSIC5=12
GMIX_SERSIC10=13

_gmix_model_dict={'full':       GMIX_FULL,
                  GMIX_FULL:    GMIX_FULL,
                  'gauss':      GMIX_GAUSS,
                  GMIX_GAUSS:   GMIX_GAUSS,
                  'turb':       GMIX_TURB,
                  GMIX_TURB:    GMIX_TURB,
                  'exp':        GMIX_EXP,
                  GMIX_EXP:     GMIX_EXP,
                  'dev':        GMIX_DEV,
                  GMIX_DEV:     GMIX_DEV,
                  'bdc':        GMIX_BDC,
                  GMIX_BDC:     GMIX_BDC,
                  'bdf':        GMIX_BDF,
                  GMIX_BDF:     GMIX_BDF,

                  GMIX_FRACDEV: GMIX_FRACDEV,
                  'fracdev': GMIX_FRACDEV,

                  GMIX_CM: GMIX_CM,
                  'cm': GMIX_CM,

                  'coellip':    GMIX_COELLIP,
                  GMIX_COELLIP: GMIX_COELLIP,

                  'sersic':    GMIX_SERSIC,
                  GMIX_SERSIC: GMIX_SERSIC,

                  'sersic5':    GMIX_SERSIC5,
                  GMIX_SERSIC5: GMIX_SERSIC5,

                  'sersic10':    GMIX_SERSIC10,
                  GMIX_SERSIC10: GMIX_SERSIC10,
                
                  'gaussmom': GMIX_GAUSSMOM,
                  GMIX_GAUSSMOM: GMIX_GAUSSMOM}

_gmix_string_dict={GMIX_FULL:'full',
                   'full':'full',
                   GMIX_GAUSS:'gauss',
                   'gauss':'gauss',
                   GMIX_TURB:'turb',
                   'turb':'turb',
                   GMIX_EXP:'exp',
                   'exp':'exp',
                   GMIX_DEV:'dev',
                   'dev':'dev',
                   GMIX_BDC:'bdc',
                   'bdc':'bdc',
                   GMIX_BDF:'bdf',
                   'bdf':'bdf',

                   GMIX_FRACDEV:'fracdev',
                   'fracdev':'fracdev',

                   GMIX_CM:'cm',
                   'cm':'cm',

                   GMIX_COELLIP:'coellip',
                   'coellip':'coellip',

                   GMIX_SERSIC:'sersic',
                   'sersic':'sersic',

                   GMIX_SERSIC5:'sersic5',
                   'sersic5':'sersic5',
                   GMIX_SERSIC10:'sersic10',
                   'sersic10':'sersic10',
                   
                   GMIX_GAUSSMOM:'gaussmom',
                   'gaussmom':'gaussmom',
                  }


_gmix_npars_dict={GMIX_GAUSS:6,
                  GMIX_TURB:6,
                  GMIX_EXP:6,
                  GMIX_DEV:6,

                  GMIX_FRACDEV:1,
                  GMIX_CM:6,

                  GMIX_BDC:8,
                  GMIX_BDF:7,
                  GMIX_SERSIC:7,
                  GMIX_SERSIC5:7,
                  GMIX_SERSIC10:7,
                  GMIX_GAUSSMOM: 6}

_gmix_ngauss_dict={GMIX_GAUSS:1,
                   'gauss':1,
                   GMIX_TURB:3,
                   'turb':3,
                   GMIX_EXP:6,
                   'exp':6,
                   GMIX_DEV:10,
                   'dev':10,

                   GMIX_FRACDEV:16,

                   GMIX_CM:16,

                   GMIX_BDC:16,
                   GMIX_BDF:16,

                   'sersic':5,
                   'sersic5':5,
                   'sersic10':10,
                   GMIX_SERSIC:5,
                   GMIX_SERSIC5:5,
                   GMIX_SERSIC10:10,

                   GMIX_GAUSSMOM: 1,

                   'em1':1,
                   'em2':2,
                   'em3':3,
                   'coellip1':1,
                   'coellip2':2,
                   'coellip3':3}


_gauss2d_dtype=[('p','f8'),
                ('row','f8'),
                ('col','f8'),
                ('irr','f8'),
                ('irc','f8'),
                ('icc','f8'),
                ('det','f8'),
                ('norm_set','i4'),
                ('drr','f8'),
                ('drc','f8'),
                ('dcc','f8'),
                ('norm','f8'),
                ('pnorm','f8')]

_cm_dtype=[('fracdev','f8'),
                  ('TdByTe','f8'), # ratio Tdev/Texp
                  ('Tfactor','f8'),
                  ('gmix',_gauss2d_dtype,16)]

def get_model_num(model):
    """
    Get the numerical identifier for the input model,
    which could be string or number
    """
    return _gmix_model_dict[model]

def get_model_name(model):
    """
    Get the string identifier for the input model,
    which could be string or number
    """
    return _gmix_string_dict[model]

def get_model_npars(model):
    """
    Get the number of parameters for the input model,
    which could be string or number
    """
    mi=_gmix_model_dict[model]
    return _gmix_npars_dict[mi]


class GMixND(object):
    """
    Gaussian mixture in arbitrary dimensions.  A bit awkward
    in dim=1 e.g. becuase assumes means are [ndim,npars]
    """
    def __init__(self, weights=None, means=None, covars=None, file=None, rng=None):

        if rng is None:
            rng=numpy.random.RandomState()
        self.rng=rng

        if file is not None:
            self.load_mixture(file)
        else:
            if (weights is not None
                    and means is not None
                    and covars is not None):
                self.set_mixture(weights, means, covars)
            elif (weights is not None
                    or means is not None
                    or covars is not None):
                raise RuntimeError("send all or none of weights, means, covars")

    def set_mixture(self, weights, means, covars):
        """
        set the mixture elements
        """

        # copy all to avoid it getting changed under us and to
        # make sure native byte order

        weights = numpy.array(weights, dtype='f8', copy=True)
        means=numpy.array(means, dtype='f8', copy=True)
        covars=numpy.array(covars, dtype='f8', copy=True)

        if len(means.shape) == 1:
            means = means.reshape( (means.size, 1) )
        if len(covars.shape) == 1:
            covars = covars.reshape( (covars.size, 1, 1) )

        self.weights = weights
        self.means=means
        self.covars=covars



        self.ngauss = self.weights.size

        sh=means.shape
        if len(sh) == 1:
            raise ValueError("means must be 2-d even for ndim=1")

        self.ndim = sh[1]

        self._calc_icovars_and_norms()

        self.tmp_lnprob = zeros(self.ngauss)

    def fit(self, data, ngauss, n_iter=5000, min_covar=1.0e-6,
            doplot=False, **keys):
        """
        data is shape
            [npoints, ndim]
        """
        from sklearn.mixture import GaussianMixture

        if len(data.shape) == 1:
            data = data[:,numpy.newaxis]

        print("ngauss:   ",ngauss)
        print("n_iter:   ",n_iter)
        print("min_covar:",min_covar)

        gmm=GaussianMixture(
            n_components=ngauss,
            max_iter=n_iter,
            reg_covar=min_covar,
            covariance_type='full',
        )

        gmm.fit(data)

        if not gmm.converged_:
            print("DID NOT CONVERGE")

        self._gmm=gmm
        self.set_mixture(gmm.weights_, gmm.means_, gmm.covariances_)

        if doplot:
            plt=self.plot_components(data=data,**keys)
            return plt


    def save_mixture(self, fname):
        """
        save the mixture to a file
        """
        import fitsio

        print("writing gaussian mixture to :",fname)
        with fitsio.FITS(fname,'rw',clobber=True) as fits:
            fits.write(self.weights, extname='weights')
            fits.write(self.means, extname='means')
            fits.write(self.covars, extname='covars')
        
    def load_mixture(self, fname):
        """
        load the mixture from a file
        """
        import fitsio

        print("loading gaussian mixture from:",fname)
        with fitsio.FITS(fname) as fits:
            weights = fits['weights'].read()
            means = fits['means'].read()
            covars = fits['covars'].read()
        self.set_mixture(weights, means, covars)

    def get_lnprob_scalar(self, pars_in):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=1
        #pars=numpy.asanyarray(pars_in, dtype='f8')
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        lnp=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                         self.means,
                                         self.icovars,
                                         self.tmp_lnprob,
                                         pars,
                                         None,
                                         dolog)
        return lnp

    def get_prob_scalar(self, pars_in):
        """
        (x-xmean) icovar (x-xmean)
        """
        dolog=0
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        p=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                       self.means,
                                       self.icovars,
                                       self.tmp_lnprob,
                                       pars,
                                       None,
                                       dolog)
        return p


    def get_lnprob_array(self, pars):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        lnp=zeros(n)

        for i in xrange(n):
            lnp[i] = self.get_lnprob_scalar(pars[i,:])

        return lnp

    def get_prob_array(self, pars):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        p=zeros(n)

        for i in xrange(n):
            p[i] = self.get_prob_scalar(pars[i,:])

        return p

    def get_prob_scalar_sub(self, pars_in, use=None):
        """
        Only include certain components
        """

        if use is not None:
            use=numpy.array(use,dtype='i4',copy=False)
            assert use.size==self.ngauss

        dolog=0
        pars=numpy.array(pars_in, dtype='f8', ndmin=1, order='C')
        p=_gmix.gmixnd_get_prob_scalar(self.log_pnorms,
                                       self.means,
                                       self.icovars,
                                       self.tmp_lnprob,
                                       pars,
                                       use,
                                       dolog)
        return p

    def get_prob_array_sub(self, pars, use=None):
        """
        array input
        """

        if len(pars.shape) == 1:
            pars = pars[:,numpy.newaxis]

        n=pars.shape[0]
        p=zeros(n)

        for i in xrange(n):
            p[i] = self.get_prob_scalar_sub(pars[i,:], use=use)

        return p


    def sample(self, n=None):
        """
        sample from the gaussian mixture
        """
        if not hasattr(self, '_gmm'):
            self._set_gmm()

        if n is None:
            is_one=True
            n=1
        else:
            is_one=False

        samples,labels = self._gmm.sample(n)

        if self.ndim==1:
            samples = samples[:,0]

        if is_one:
            samples = samples[0]

        return samples

    def _make_gmm(self, ngauss):
        """
        Make a GMM object for sampling
        """
        from sklearn.mixture import GaussianMixture

        gmm=GaussianMixture(
            n_components=ngauss,
            max_iter=10000,
            reg_covar=1.0e-12,
            covariance_type='full',
            random_state=self.rng,
        )

        return gmm


    def _set_gmm(self):
        """
        Make a GMM object for sampling
        """
        import sklearn.mixture

        # these numbers are not used because we set the means, etc by hand
        ngauss=self.weights.size

        gmm = self._make_gmm(ngauss)
        gmm.means_ = self.means.copy()
        #gmm.covars_ = self.covars.copy()
        gmm.covariances_ = self.covars.copy()
        gmm.weights_ = self.weights.copy()

        gmm.precisions_cholesky_ = sklearn.mixture.gaussian_mixture._compute_precision_cholesky(
            self.covars, 'full',
        )

        self._gmm=gmm 

    def _calc_icovars_and_norms(self):
        """
        Calculate the normalizations and inverse covariance matrices
        """
        from numpy import pi

        twopi = 2.0*pi

        #if self.ndim==1:
        if False:
            norms = 1.0/sqrt(twopi*self.covars)
            icovars = 1.0/self.covars
        else:
            norms = zeros(self.ngauss)
            icovars = zeros( (self.ngauss, self.ndim, self.ndim) )
            for i in xrange(self.ngauss):
                cov = self.covars[i,:,:]
                icov = numpy.linalg.inv( cov )

                det = numpy.linalg.det(cov)
                n=1.0/sqrt( twopi**self.ndim * det )

                norms[i] = n
                icovars[i,:,:] = icov

        self.norms = norms
        self.pnorms = norms*self.weights
        self.log_pnorms = log(self.pnorms)
        self.icovars = icovars

    def plot_components(self, data=None, **keys):
        """
        """
        import biggles 
        import pcolors

        # first for 1d,then generalize
        assert self.ndim==1

        # draw a random sample to get a feel for the range

        xmin=keys.pop('min',None)
        xmax=keys.pop('max',None)
        if data is not None:
            if xmin is None:
                xmin=data.min()
            if xmax is None:
                xmax=data.max()
        else:
            r = self.sample(100000)
            if xmin is None:
                xmin=r.min()
            if xmax is None:
                xmax=r.max()

        x = numpy.linspace(xmin, xmax, num=10000)

        ytot = self.get_prob_array(x)
        ymax=ytot.max()
        ymin=1.0e-6*ymax
        ytot=ytot.clip(min=ymin)

        xrng=keys.pop('xrange',None) 
        yrng=keys.pop('yrange',None) 
        if xrng is None:
            xrng=[xmin,xmax]
        if yrng is None:
            yrng=[ymin,1.1*ymax]



        if data is not None:
            binsize=keys.pop('binsize',None)
            if binsize is None:
                binsize=0.1*data.std()

            histc = biggles.make_histc(
                data,
                min=xmin,
                max=xmax,
                yrange=yrng,
                binsize=binsize,
                color='orange',
                width=3,
                norm=1,
            )
            loghistc = biggles.make_histc(
                data,
                min=xmin,
                max=xmax,
                ylog=True,
                yrange=yrng,
                binsize=binsize,
                color='orange',
                width=3,
                norm=1,
            )
        else:
            histc=None
            loghistc=None




        plt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            aspect_ratio=1.0/1.618,
            xrange=xrng,
            yrange=yrng,
        )
        logplt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            ylog=True,
            xrange=xrng,
            yrange=yrng,
            aspect_ratio=1.0/1.618,
        )

        curves = []
        logcurves = []
        if histc is not None:        
            curves.append(histc)
            logcurves.append(loghistc)

        ctot = biggles.Curve(x, ytot, type='solid', color='black')
        curves.append(ctot)
        logcurves.append(ctot)



        colors = pcolors.rainbow(self.ngauss)
        use = numpy.zeros(self.ngauss, dtype='i4')
        for i in xrange(self.ngauss):

            use[i] = 1

            y = self.get_prob_array_sub(x, use=use)
            y = y.clip(min=ymin)


            c = biggles.Curve(x, y, type='solid', color=colors[i])
            curves.append(c)
            logcurves.append(c)

            use[:]=0

        ymax=ytot.max()

        plt.add(*curves)
        logplt.add(*logcurves)

        tab = biggles.Table(2,1)
        tab[0,0] = plt
        tab[1,0] = logplt
        show=keys.pop('show',False)
        if show:
            width=keys.pop('width',1000)
            height=keys.pop('height',1000)
            tab.show(width=width, height=height, **keys)
        return tab

    def plot_components_old(self, **keys):
        """
        """
        import biggles 
        import pcolors

        # first for 1d,then generalize
        assert self.ndim==1

        # draw a random sample to get a feel for the range
        r = self.sample(100000)

        xmin,xmax=r.min(),r.max()

        x = numpy.linspace(xmin, xmax, num=1000)

        ytot = self.get_prob_array(x)
        ymax=ytot.max()
        ymin=1.0e-6*ymax
        ytot=ytot.clip(min=ymin)

        plt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            aspect_ratio=1.0/1.618,
        )
        logplt=biggles.FramedPlot(
            xlabel='x',
            ylabel='P(x)',
            ylog=True,
            aspect_ratio=1.0/1.618,
        )


        ctot = biggles.Curve(x, ytot, type='solid', color='black')

        curves = [ctot]

        ytot2 = ytot*0

        colors = pcolors.rainbow(self.ngauss)
        use = numpy.zeros(self.ngauss, dtype='i4')
        for i in xrange(self.ngauss):

            use[i] = 1

            y = self.get_prob_array_sub(x, use=use)
            y = y.clip(min=ymin)

            ytot2 += y

            c = biggles.Curve(x, y, type='solid', color=colors[i])
            curves.append(c)

            use[:]=0

        ymax=ytot.max()

        ctot2 = biggles.Curve(x, ytot2, type='dashed', color='red')
        curves.append(ctot2)
        plt.add(*curves)
        logplt.add(*curves)

        tab = biggles.Table(2,1)
        tab[0,0] = plt
        tab[1,0] = logplt
        show=keys.pop('show',False)
        if show:
            tab.show(**keys)
        return tab



_moms_flagmap={
    0:'ok',
    1:'zero weight encountered',
}


