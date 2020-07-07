import numpy as np
import math
from astropy.convolution import Gaussian2DKernel, Kernel2D, Moffat2DKernel
from astropy.modeling import models
from astropy.constants import h,c,k_B
import matplotlib.pyplot as plt
import scipy
from scipy.interpolate import interp2d
from math import floor,ceil

h = h.value
c = c.value
k_B = k_B.value

def _round_up_to_odd_integer(value):
    i = math.ceil(value)
    if i % 2 == 0:
        return i + 1
    else:
        return i

class Moffat2DKernel2(Kernel2D):

    _is_bool = False

    def __init__(self, amplitude, x_0, y_0, gamma, alpha, **kwargs):
        self._model = models.Moffat2D(amplitude, x_0, y_0, gamma, alpha)
        self._default_size = _round_up_to_odd_integer(4.0 * self._model.fwhm)
        super().__init__(**kwargs)
        self._truncation = None

def blackbody(l,T=10000):
    '''
    Inputs
    ------
    l in angstrom
    T in kelvin
    '''
    l = l*1e-10 # converting to meters
    bb = 2*c/l**4*(1/(np.exp(h*c/(l*k_B*T))-1))
    bb /= bb.sum()
    return bb

def moffat(seeing, amplitude = None, center = (0,0),mode=None,factor=None, xsize=None,ysize=None):
    alpha = 3.5
    gamma = 0.5 * seeing / np.sqrt(2**(1./alpha) - 1.)
    if amplitude == None:
        amplitude = (alpha - 1.0) / (np.pi * gamma * gamma)
    morph = Moffat2DKernel2(amplitude,
                            center[0],
                            center[1],
                            gamma,
                            alpha,
                            x_size=xsize,
                            y_size=ysize,
                            mode=mode,
                            factor=factor)
    return morph
class BlackbodyCubeSampler(object):
    def __init__(self,wl,sizex,sizey,size=None,seeing=3,oversampling=20.0):
        self.wl = wl
        self.seeing = seeing
        if size is None:
            morph = moffat(seeing/0.75*oversampling, center=(0,0), mode='center')
            self.size = morph._default_size / (oversampling*2)
        else:
            self.size = size
            morph = moffat(seeing/0.75*oversampling, center=(0,0), mode='center',size=size*2*oversampling)
        size = self.size
        morph = morph.array
        x = np.linspace(-size,size,size*2*oversampling)
        y = np.linspace(-size,size,size*2*oversampling)
        f = interp2d(x,y,morph)
        self.size = round(self.size)
        self.sizex = sizex
        self.sizey = sizey
        self._f = f
        self.mod  = np.zeros((sizex,sizey))
    def sample(self,T,A,x0,y0):
        size = self.size
        sizex = self.sizex
        sizey = self.sizey
        minx = max(-size/2., ceil(2*(- x0 - sizex/2))/2)
        maxx = min( size/2., floor(2*(- x0 + sizex/2))/2)
        miny = max(-size/2., ceil(2*(- y0 - sizey/2))/2)
        maxy = min( size/2., floor(2*(- y0 + sizey/2))/2)
        x = np.arange(minx+1/2,maxx+1/2,1)
        y = np.arange(miny+1/2,maxy+1/2,1)
        spectra = A*blackbody(self.wl, T)
        dx0 = x0-round(x0*2)/2
        dy0 = y0-round(y0*2)/2
        morph = self._f(x-dx0,y-dy0)
        maxy, miny = int(min(sizey, round(y0 + sizey/2 + size/2))), int(max(0, round(y0 + sizey/2 - size/2)))
        maxx, minx = int(min(sizex, round(x0 + sizex/2 + size/2))), int(max(0, round(x0 + sizex/2 - size/2)))
        self.mod[miny:maxy,minx:maxx] = morph
        #cube = np.outer(spectra.T, morph)
        #cube.shape = (len(spectra), morph.shape[0], morph.shape[1])
        cube = np.outer(spectra.T, self.mod)
        cube.shape = (len(spectra), sizey, sizex)

        return cube

class BlackbodyCubeSampler_whole(object):
    def __init__(self,wl,xsize,ysize=None,seeing=3,oversampling=20.0):
        self.wl = wl
        self.seeing = seeing
        if ysize == None:
            ysize = xsize
        self.xsize = xsize
        self.ysize = ysize
        morph = moffat(seeing/0.75*oversampling,
                       center=(0,0), mode='center',
                       xsize=xsize*2*oversampling,
                       ysize=ysize*2*oversampling)
        morph = morph.array
        #import pdb; pdb.set_trace()
        x = np.arange(-xsize, xsize, 1/oversampling) + 1/(2*oversampling)
        y = np.arange(-ysize, ysize, 1/oversampling) + 1/(2*oversampling)
        f = interp2d(x,y,morph,kind='linear')
        self._f = f
        self.x = np.arange(-xsize/2,xsize/2,1) + 1/2
        self.y = np.arange(-ysize/2,ysize/2,1) + 1/2
    def sample(self,T,A,x0,y0):
        spectra = A*blackbody(self.wl, T)
        #import pdb; pdb.set_trace()
        morph = self._f(self.x-x0,self.y-y0)

        cube = np.outer(spectra.T, morph)
        cube.shape = (len(spectra), morph.shape[0], morph.shape[1])
        return cube
def blackbodycube(T, A, x, y, wl, size, seeing):
    spectra = A*blackbody(wl, T)
    morph = moffat(seeing*1/0.75, center=(x,y), mode='oversample', factor= 20, xsize=size,ysize=size)
    cube = np.outer(spectra.T, morph)
    cube.shape = (len(spectra), morph.shape[0], morph.shape[1])
    return cube

