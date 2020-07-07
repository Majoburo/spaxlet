import numpy as np
import matplotlib.pyplot as plt
import bilby
from astropy.constants import h,c,k_B
from synthetic_cube import BlackbodyCubeSampler,blackbodycube

class BlackBody(bilby.Likelihood):
    def __init__(self, wl, images, weights, seeing):
        super().__init__(parameters={
#            'T': None,
#            'A': None,
            'x': None,
            'y': None,
            })
        self.wl = wl
        self.images = images
        self.stds = 1/weights**0.5
        #weights[weights<1e-20]=1e-20
        self.seeing = seeing
        _,sizex,sizey = images.shape
        self.loglnorm = -0.5*np.sum(np.log(2*np.pi*self.stds**2))
        self.bcs = BlackbodyCubeSampler(wl, sizex, sizey, seeing=seeing)
    def log_likelihood(self):
        T = 5600 #self.parameters['T']
        A = 400 #self.parameters['A']
        x = self.parameters['x']
        y = self.parameters['y']
        images = self.images
        stds = self.stds
        #_, sizex, sizey = images.shape
        #maxy, miny = int(min(sizey, round(y + sizey/2 + bcs.ysize/2))), int(max(0, round(y + sizey/2 - bcs.ysize/2)))
        #maxx, minx = int(min(sizex, round(x + sizex/2 + bcs.xsize/2))), int(max(0, round(x + sizex/2 - bcs.xsize/2)))
        #cimages = images[:, miny:maxy, minx:maxx]
        #cstds = stds[:, miny:maxy, minx:maxx]
        #cstds = 1e-3*np.ones_like(images) #stds[:, miny:maxy, minx:maxx]
        bcss = self.bcs.sample(T,A,x,y)
        #bcss = blackbodycube(T, A, x, y, self.wl, images.shape[1], self.seeing)
        res = (images - bcss)/stds
        #plt.imshow(images.sum(axis=0))
        #plt.show()
        #plt.imshow(cimages.sum(axis=0))
        #plt.show()
        #plt.imshow(bcss.sum(axis=0))
        #plt.show()
        #l1 = (1-pb)/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - m*x - b)**2 / 2 / sy**2)
        #l2 = pb/np.sqrt(2*np.pi*(vb+sy**2))* np.exp(-(y - yb)**2 / 2 / (vb+sy**2))
        #l = 1/np.sqrt(2*np.pi*sy**2) * np.exp(-(y - A*blackbody(x,T))**2 / 2 / sy**2)
        #res = (images - blackbodycube(T, A, x, y, wl, size, seeing))/stds
        #return -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * stds ** 2))
        #logl = -0.5 * np.sum(res ** 2 + np.log(2 * np.pi * stds ** 2))
        logl = -0.5*np.sum(res ** 2) + self.loglnorm
        #logl += np.log(np.sum(cimages))
        #print(x,y,logl)
        #print(logl)
        plt.scatter(x+63/2,y+63/2,c="red",alpha=1-np.exp(logl))
        plt.draw()
        return logl
weights = 1/np.load('stds.npy')**2
size = weights.shape[1]
wl = np.load('wl.npy')
images = np.load('images.npy')
seeing = 2

def test_log_likelihood(args):
    T,A,x,y = args
    _, sizex, sizey = images.shape
    bcs = BlackbodyCubeSampler(wl, sizex, sizey, seeing=seeing)
    #stds = 1/weights**0.5
    cstds = 1e-3*np.ones_like(images) #stds[:, miny:maxy, minx:maxx]
    bcss = bcs.sample(T,A,x,y)
    res = (images - bcss)/cstds
    plt.imshow(images.sum(axis=0))
    plt.show()
    plt.imshow(bcss.sum(axis=0))
    plt.show()
    plt.imshow(res.sum(axis=0))
    plt.show()
    logl = -0.5 * np.sum(res ** 2) + -0.5*np.sum(np.log(2 * np.pi * cstds ** 2))
    return logl
plt.ion()
plt.imshow(images.sum(axis=0))
likelihood = BlackBody(wl, images, weights, seeing)
uniform = bilby.core.prior.Uniform
priors = dict(
#        T=uniform(name='T',minimum=1e3,maximum=1e4),
#        A=uniform(name='A',minimum=0,maximum=1e3),
        x=uniform(name='x',minimum=-size/2+2,maximum=size/2-2),
        y=uniform(name='y',minimum=-size/2+2,maximum=size/2-2),
        #x=uniform(name='x',minimum=-size/2,maximum=-29),
        #y=uniform(name='y',minimum=-3,maximum=3),
        )
result = bilby.run_sampler(likelihood=likelihood,priors=priors,
        sampler='dynesty',npoints=500,walks=10,outdir='samples',label='bbtest',dlogz=10.0,plot=True)
result.plot_corner()
