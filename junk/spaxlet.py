import numpy as np
import argparse
import scarlet
import scarlet.display
import matplotlib
import matplotlib.pyplot as plt
from scarlet.color_tools import wavelength_to_rgb as spectrum_to_rgb
from scarlet.color_tools import spectrum_to_rgb as real_spectrum_to_rgb
from astropy.io import fits
from astropy.nddata.utils import Cutout2D
import astropy.wcs as wcs
from ps_query import query
from astropy.coordinates import SkyCoord
import os
import pandas as pd
import glob
from functools import partial
matplotlib.rc('image', cmap='inferno', interpolation='none', origin='lower')
basename = ""
sextractor = True

def cleandata(chdu,ehdu):
    """
    Input
    -----
    chdu: astropy.io.fits object
        data cube
    ehdu: astropy.io.fits object
        variance cube
    w: astropy.wcs.WCS object
        wcs object
    Output
    ------
    images: ndarray
        data array
    weights: ndarray
        weights array
    ifu_wl: array
        wl array
    """

    # Get wl range
    wlstart = chdu[0].header['CRVAL3']
    wldelta = chdu[0].header['CDELT3']
    wlend = wlstart + wldelta*chdu[0].header['NAXIS3']
    ifu_wl = np.arange(wlstart,wlend,wldelta)

    # Clean up nans, infs and all nan in a wl bin
    images = chdu[0].data
    weights = 1/ehdu[0].data
    weights[np.isnan(weights)] = 0.
    weights[np.isinf(weights)] = 0.
    weights[np.isinf(images)] = 0.
    wmask = (weights!=0).sum((1,2))!=0

    images = images[wmask]
    weights = weights[wmask]
    ifu_wl = ifu_wl[wmask]

    return images, weights, ifu_wl, wmask

def query_ps_from_wcs(w):
    """Query PanStarrs for a wcs.
    """
    assert w.axis_type_names == ['RA', 'DEC', 'pixel']

    nra,ndec = w.array_shape[1:]
    dra,ddec = w.wcs.cdelt[:2]
    c = wcs.utils.pixel_to_skycoord(nra/2.,ndec/2.,w)
    ddeg = np.linalg.norm([dra*nra/2,ddec*ndec/2])
    pd_table = query(c.ra.value,c.dec.value,ddeg)

    # Crop sources to those in the cube limits
    scat = wcs.utils.skycoord_to_pixel(
        SkyCoord(pd_table['raMean'],pd_table['decMean'], unit="deg"),
        w,
        origin=0,
        mode='all'
    )
    mask = (scat[0] < nra+1)*(scat[1] < ndec+1)*(scat[0] > -1)*(scat[1] > -1)
    pd_table = pd_table[mask]
    pd_table['x'] = scat[0][mask]
    pd_table['y'] = scat[1][mask]

    return pd_table


def select_sources(cube,ifu_wl,pd_table,stretch = 100, Q = 5, minimum = 0):
    from scarlet.display import AsinhMapping

    ifu_rgb = np.array([ spectrum_to_rgb(wl) for wl in ifu_wl]).T
    norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
    img_rgb = scarlet.display.img_to_rgb(cube, channel_map=ifu_rgb, norm=norm)
    #img_rgb = scarlet.display.cube_to_rgb(cube, ifu_wl, norm=norm)
    from pointpicker import pointpicker
    #plt.imshow(cube.sum(axis=0))
    #img_rgb = cube.sum(axis=0)
    if sextractor:
        sexx, sexy = identify_sources(pd_table.name, plot=False)
        sexpts = np.array(list(zip(sexx,sexy)))
        srcdic = pointpicker(img_rgb, sexpts, pd_table[['x','y']].to_numpy())
        for srctype, indexes in srcdic.items():
            srcdic[srctype] = sexpts[indexes]
    else:
        from pointshifter import pointshifter
        shiftedpoints = pointshifter(img_rgb,pd_table[['x','y']].to_numpy())
        pd_table.x = shiftedpoints[:,0]
        pd_table.y = shiftedpoints[:,1]
        srcdic = pointpicker(img_rgb,pd_table[['x','y']].to_numpy())
    return srcdic

def identify_sources(fitsfile, plot=True):
    os.chdir("../config")
    os.system("sextractor "+ fitsfile)
    sexdata = np.loadtxt('test.cat')
    os.chdir("../junk")
    xsrc = sexdata[:,6]-1
    ysrc = sexdata[:,7]-1
    if plot:
        h = fits.open(fitsfile)
        fig,ax = plt.subplots(1,2)
        ax[0].imshow(np.log(h[0].data))
        ax[1].imshow(np.log(h[0].data))
        ax[1].scatter(xsrc,ysrc,color="r")
        plt.show()

    return xsrc, ysrc


def define_model(images,weights,psf="moffatpsf.npy"):
    """ Create model psf and obsevation
    """
    start_psf = np.load(psf)
    out = np.outer(np.ones(len(images)),start_psf)
    # WARNING, using same arbitray psf for all now.
    out.shape = (len(images),start_psf.shape[0],start_psf.shape[1])
    psfs = scarlet.PSF(out)
    model_psf = scarlet.PSF(partial(scarlet.psf.gaussian, sigma=.8),
                            shape=(None, 8, 8))
    #model_psf = scarlet.PSF(partial(scarlet.psf.moffat),
    #                        shape=(None, 8, 8))
    model_frame = scarlet.Frame(
                  images.shape,
                  psfs=model_psf)
    observation = scarlet.Observation(
                  images,
                  weights=weights,
                  psfs=psfs).match(model_frame)
    return model_frame, observation

def define_sources(srcdic,model_frame,observation):
    """
    """
    sources = []

    for key, loc in srcdic.items():
        if key == 'point':
            for x,y in loc:
                new_source = scarlet.PointSource(model_frame,
                                                 (y, x),
                                                 observation)
                sources.append(new_source)
        elif key == 'extended':
            for x,y in loc:
                new_source = scarlet.ExtendedSource(model_frame,
                                                    (y, x),
                                                    observation,
                                                    shifting=True)
                sources.append(new_source)
        elif key == 'multi':
            for x,y in loc:
                new_source = scarlet.MultiComponentSource(model_frame,
                                                          (y, x),
                                                          observation,
                                                          shifting=True)
                sources.append(new_source)
    return sources

def blend(sources, observation):
    blend = scarlet.Blend(sources, observation)
    blend.fit(200,1e-7)
    print("scarlet ran for {0} iterations to logL = {1}".format(len(blend.loss), -blend.loss[-1]))
    plt.plot(-np.array(blend.loss))
    plt.xlabel('Iteration')
    plt.ylabel('log-Likelihood')
    plt.savefig("loglikehood"+basename+".pdf")
    plt.close()
    return blend


def parseargs(fname=None):
    parser = argparse.ArgumentParser(description="Extract sources in spectral cube.")
    parser.add_argument("cubes", type=str, nargs='+', help="cube(s) you want to process")
    if fname is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args([fname])
    return args

def main(fname=None):
    global basename

    args = parseargs(fname)
    #import pdb;pdb.set_trace()
    for cube in args.cubes:
        path,file = os.path.split(cube)
        chdu = fits.open(cube)
        if chdu[0].header["INSTRUME"] == 'virus':
            basename = file.split("_cube")[0]
            ecube = path + "/" + basename + "_error_cube.fits"
            fcube = path + "/" + basename + ".fits"
            ccube = path + "/" + basename + "cutout.fits"
            hdu = fits.open(fcube)
            w = wcs.WCS(hdu[0].header, hdu)
            cutout1 = Cutout2D(hdu[0].data, (hdu[0].shape[0]/2.-1,hdu[0].shape[1]/2.-1),63,w)
            w = cutout1.wcs
            chdu = fits.open(cube)
            ehdu = fits.open(ecube)
            chdu[0].header.update(w.to_header())
            #chdu.writeto(cube)
            w = wcs.WCS(chdu[0].header, chdu)

        if chdu[0].header["INSTRUME"] == 'lrs2':
            print("No error cubes yet for lrs2. Exiting.")
            exit()

        if chdu[0].header["INSTRUME"] == 'injection':
            basename = file.split("_cube")[0]
            ecube = path + "/" + basename + "_error_cube.fits"
            #For now cause i need to sleep...
            ecube = "/home/majoburo/Work/GWHET-paper/cubes/20191214_0000023_047_error_cube.fits"
            ccube = path + "/" + basename + "cutout.fits"
            chdu = fits.open(cube)
            ehdu = fits.open(ecube)
            hdu = chdu.__deepcopy__()
            w = wcs.WCS(chdu[0].header, chdu)
        pd_table = query_ps_from_wcs(w)
        images, weights, ifu_wl, wmask = cleandata(chdu,ehdu)
        np.savez(path+"/"+basename+"_cleandata.npz",images=images, weights=weights, ifu_wl=ifu_wl, wmask=wmask)
        hdu[0].data = (images*weights).sum(axis=0)
        hdu[0].header.update(w.to_header())
        hdu.writeto(ccube,overwrite=True)
        pd_table.name = ccube
        #srcdic = select_sources(images*weights, ifu_wl, pd_table)
        #print(srcdic)
        srcdic = {'point': np.array([]), 'extended': np.array([[31.7493, 13.2891],
       [ 0.6265,  1.7569],
       [32.9087,  1.9377],
       [38.2521, 49.576 ],
       [58.1416, 39.1723],
       [28.4111, 56.7858],
       [55.5849, 25.899 ],
       [53.328 , 57.3259],
       [ 3.9251, 53.5119]]), 'multi': np.array([[27.941 , 26.3807]])}

        if not sextractor:
            for srctype, indexes in srcdic.items():
                srcdic[srctype] = pd_table.iloc[indexes][['x','y']].to_numpy()
        #seeing = chdu[0].header['VSEEING']
        model_frame, observation = define_model(images,weights)
        sources = define_sources(srcdic,model_frame,observation)
        Blend = blend(sources,observation)
        model = Blend.get_model()
        model_ = observation.render(model)

        ifu_rgb = np.array([ spectrum_to_rgb(wl) for wl in ifu_wl]).T
        stretch = 100
        Q = 5
        minimum = 0
        from scarlet.display import AsinhMapping
        norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
        scarlet.display.show_spectral_sources(sources,
                                 observation=observation,
                                 show_rendered=True,
                                 channel_map = ifu_rgb,
                                 norm=norm,
                                 show_observed=True)
        plt.savefig(path + "/" + basename + ".png")
        fp = open(path+"/"+basename+".sca", "wb")
        import pickle
        pickle.dump(sources, fp)
        fp.close()
        ##second pass
        ##import pdb; pdb.set_trace()
        #residual = images - model_
        #hdu[0].data = residual
        #rcube = path+"/res"+basename+".fits"
        #hdu[0].writeto(rcube,overwrite=True)
        #pd_table.name = rcube
        #srcdicres = select_sources(residual*weights, ifu_wl, pd_table)
        #model_frameres, observationres = define_model(residual,weights)
        #sourcesres = define_sources(srcdicres,model_frameres,observationres)
        #Blend = blend(sourcesres,observationres)
        #model = Blend.get_model()
        #model_ = observationres.render(model)
        #ifu_rgb = np.array([ spectrum_to_rgb(wl) for wl in ifu_wl]).T
        #stretch = 100
        #Q = 5
        #minimum = 0
        #from scarlet.display import AsinhMapping
        #norm = AsinhMapping(minimum=minimum, stretch=stretch, Q=Q)
        #scarlet.display.show_spectral_sources(sourcesres,
        #                         observation=observationres,
        #                         show_rendered=True,
        #                         channel_map = ifu_rgb,
        #                         norm=norm,
        #                         show_observed=True)
        #plt.savefig(path + "/res" + basename + ".png")
        ##fpres = open(path+"/"+basename+"res.sca", "wb")
        ##pickle.dump(sourcesres, fpres)
        ##fpres.close()
        aa = Blend.sources[-1].components[-2].morph
        xv,yv = np.mgrid[0:aa.shape[0],0:aa.shape[1]]
        com = (np.sum(aa*xv)/np.sum(aa)+1/2,np.sum(aa*yv)/np.sum(aa)+1/2)
        return com
if __name__ == "__main__":
    main()

