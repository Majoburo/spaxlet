from __future__ import print_function, division
import numpy as np
from functools import partial

import proxmin
from proxmin.nmf import Steps_AS

import logging
logger = logging.getLogger("scarlet")

class Blend(object):
    """The blended scene as interpreted by the deblender.
    """
    def __init__(self, sources):
        assert len(sources)
        # store all source and make search structures
        self._register_sources(sources)

        # collect all proxs_g and Ls: first A, then S
        self._proxs_g = [source.proxs_g[0] for source in self.sources] + [source.proxs_g[1] for source in self.sources]
        self._Ls = [source.Ls[0] for source in self.sources] + [source.Ls[1] for source in self.sources]

        # center update parameters
        self.center_min_dist = 1e-3
        self.center_wait = 10
        self.center_skip = 10

    def source_of(self, k):
        return self._source_of[k]

    def component_of(self, m, l):
        # search for k that has this (m,l), inverse of source_of
        for k in range(self.K):
            if self._source_of[k] == (m,l):
                return k
        raise IndexError

    def __len__(self):
        """Number of distinct sources"""
        return self.M

    @property
    def B(self):
        try:
            return self._img.shape[0]
        except AttributeError:
            return 0

    def fit(self, img, weights=None, sky=None, init_sources=True, update_order=None, e_rel=1e-2, max_iter=200):

        # set data/weights to define objective function gradients
        self.set_data(img, weights=weights, sky=sky, init_sources=init_sources, update_order=update_order, e_rel=e_rel)

        # perform up to max_iter steps
        return self.step(max_iter=max_iter)

    def step(self, max_iter=1):
        # collect all SEDs and morphologies, plus associated errors
        XA = []
        XS = []
        for k in range(self.K):
            m,l = self.source_of(k)
            XA.append(self.sources[m].sed[l])
            XS.append(self.sources[m].morph[l])
        X = XA + XS

        # update_order for bSDMM is over *all* components
        if self.update_order[0] == 0:
            _update_order = range(2*self.K)
        else:
            _update_order = range(self.K,2*self.K) + range(self.K)

        # run bSDMM on all SEDs and morphologies
        steps_g = None
        steps_g_update = 'steps_f'
        traceback = False
        accelerated = True
        res = proxmin.algorithms.bsdmm(X, self._prox_f, self._steps_f, self._proxs_g, steps_g=steps_g, Ls=self._Ls, update_order=_update_order, steps_g_update=steps_g_update, max_iter=max_iter, e_rel=self.e_rel, e_abs=self.e_abs, accelerated=accelerated, traceback=traceback)

        return self

    def set_data(self, img, weights=None, sky=None, init_sources=True, update_order=None, e_rel=1e-2, slack=0.9):
        self.it = 0
        self._model_it = -1

        if sky is None:
            self._ = img
        else:
            self._img = img-sky

        if update_order is None:
            self.update_order = [1,0] # S then A
        else:
            self.update_order = update_order

        self._set_weights(weights)
        WAmax = np.max(self._weights[1])
        WSmax = np.max(self._weights[2])
        self._stepAS = Steps_AS(WAmax=WAmax, WSmax=WSmax, slack=slack, update_order=self.update_order)
        self.step_AS = [None] * 2

        if init_sources:
            self.init_sources()

        # set sparsity cutoff for morph based on the error level
        # TODO: Computation only correct if psf=None!
        self.e_rel = [e_rel] * 2*self.K
        self.e_abs = [e_rel / self.B] * self.K + [0.] * self.K
        if weights is not None:
            for m in range(self.M):
                morph_std = self.sources[m].set_morph_sparsity(weights)
                for l in range(self.sources[m].K):
                    self.e_abs[self.K + self.component_of(m,l)] = e_rel * morph_std[l]

    def init_sources(self):
        for m in range(self.M):
            self.sources[m].init_source(self._img, weights=self._weights[0])

    def get_model(self, m=None, combine=True, combine_source_components=True):
        """Compute the current model for the entire image
        """
        if m is not None:
            source = self.sources[m]
            model = source.get_model(combine=combine_source_components)
            model_slice = source.get_slice_for(self._img.shape)
            if combine_source_components:
                model_img = np.zeros(self._img.shape)
                model_img[source.bb] = model[model_slice]
            else:
                model_img = np.zeros((source.K,) + (self._img.shape))
                for k in range(source.K):
                    model_img[k][source.bb] = model[k][model_slice]
            return model_img

        # for all sources
        if combine:
            return np.sum([self.get_model(m=m, combine_source_components=True) for m in range(self.M)], axis=0)
        else:
            models = [self.get_model(m=m, combine_source_components=combine_source_components) for m in range(self.M)]
            return np.vstack(models)

    def _register_sources(self, sources):
        self.sources = sources # do not copy!
        self.M = len(self.sources)
        self.K =  sum([source.K for source in self.sources])
        self.psf_per_band = not hasattr(sources[0].Gamma, 'shape')

        # lookup of source/component tuple given component number k
        self._source_of = []
        self.update_centers = False
        for m in range(self.M):
            self.update_centers |= bool(self.sources[m].shift_center)
            for l in range(self.sources[m].K):
                self._source_of.append((m,l))

    def _set_weights(self, weights):
        if weights is None:
            self._weights = [1,1,1]
        else:
            self._weights = [weights, None, None] # [W, WA, WS]

            # for S update: normalize the per-pixel variation
            # i.e. in every pixel: utilize the bands with large weights
            # CAVEAT: need to filter out pixels that are saturated in every band
            norm_pixel = np.median(weights, axis=0)
            mask = norm_pixel > 0
            self._weights[2] = weights.copy()
            self._weights[2][:,mask] /= norm_pixel[mask]

            # reverse is true for A update: for each band, use the pixels that
            # have the largest weights
            norm_band = np.median(weights, axis=(1,2))
            # CAVEAT: some regions may have one band missing completely
            mask = norm_band > 0
            self._weights[1] = weights.copy()
            self._weights[1][mask] /= norm_band[mask,None,None]
            # CAVEAT: mask all pixels in which at least one band has W=0
            # these are likely saturated and their colors have large weights
            # but are incorrect due to missing bands
            mask = ~np.all(weights>0, axis=0)
            # and mask all bands for that pixel:
            # when estimating A do not use (partially) saturated pixels
            self._weights[1][:,mask] = 0

    def _compute_model(self):
        # make sure model at current iteration is computed when needed
        # irrespective of function that needs it
        if self._model_it < self.it:
            self._models = self.get_model(combine=False, combine_source_components=False) # model each each component over image
            self._model = np.sum(self._models, axis=0)
            self._model_it = self.it

    def _prox_f(self, X, step, Xs=None, j=None):

        # which update to do now
        AorS = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A:
        # build model only once per iteration
        if k == 0:
            if AorS == self.update_order[0]:
                # TODO: check source size
                # self.resize_sources()
                self._compute_model()

                # update positions?
                if self.update_centers:
                    if self.it >= self.center_wait and self.it % self.center_skip == 0:
                        self._update_positions()
                self.it += 1

            # compute weighted residuals
            self._diff = self._weights[AorS + 1]*(self._model-self._img)

        # A update
        if AorS == 0:
            m,l = self.source_of(k)
            if not self.sources[m].fix_sed[l]:
                # gradient of likelihood wrt A: nominally np.dot(diff, S^T)
                # but with PSF convolution, S_ij -> sum_q Gamma_bqi S_qj
                # however, that's exactly the operation done for models[k]
                # caveat: the model is SED * convolved model -> need to divide
                grad = np.einsum('...ij,...ij', self._diff, self._models[k] / self.sources[m].sed[l].T[:,None,None])

                # apply per component prox projection and save in source
                self.sources[m].sed[l] =  self.sources[m].prox_sed[l](X - step*grad, step)
            return self.sources[m].sed[l]

        # S update
        elif AorS == 1:
            m,l = self.source_of(k)
            if not self.sources[m].fix_morph[l]:
                # gradient of likelihood wrt S: nominally np.dot(A^T,diff)
                # but again: with convolution, it's more complicated

                # first create diff image in frame of source k
                diff_k = np.zeros(self.sources[k].shape)
                diff_k[self.sources[k].get_slice_for(self._img.shape)] = self._diff[self.sources[k].bb]

                grad = np.zeros_like(X)
                if not self.psf_per_band:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma.T.dot(diff_k[b].flatten())
                else:
                    for b in range(self.B):
                        grad += self.sources[m].sed[l,b]*self.sources[m].Gamma[b].T.dot(diff_k[b].flatten())

                # apply per component prox projection and save in source
                self.sources[m].morph[l] = self.sources[m].prox_morph[l](X - step*grad, step)
            return self.sources[m].morph[l]
        else:
            raise ValueError("Expected index j in [0,%d]" % (2*self.K))

    def _steps_f(self, j, Xs):
        # which update to do now
        AorS = j//self.K
        k = j%self.K

        # computing likelihood gradients for S and A: only once per iteration
        if AorS == self.update_order[0] and k==0:
            self._compute_model()

            # build temporary A,S matrices
            B, Ny, Nx = self._img.shape
            A = np.empty((self.B,self.K))
            for k_ in range(self.K):
                m,l = self._source_of[k_]
                A[:,k_] = self.sources[m].sed[l]
            if not self.psf_per_band:
                # model[b] is simple SED[b] * S, need to divide by SED
                b = 0
                S = self._models[:,b,:,:].reshape((self.K, Ny*Nx)) / A.T[:,b][:,None]
            else:
                # TODO: replace this with the current models, for each band
                # i.e. one S per band -> step_size per band
                raise NotImplementedError
            self.step_AS[0] = self._stepAS(0, [A, S])
            self.step_AS[1] = self._stepAS(1, [A, S])
        return self.step_AS[AorS]

    def _update_positions(self):
        # residuals weighted with full/original weight matrix
        y = self._weights[0]*(self._model-self._img)
        for m in range(self.M):
            if self.sources[m].shift_center:
                source = self.sources[m]
                bb_m = source.bb
                diff_x,diff_y = self._get_shift_differential(m)
                diff_x[:,:,-1] = 0
                diff_y[:,-1,:] = 0
                # least squares for the shifts given the model residuals
                MT = np.vstack([diff_x.flatten(), diff_y.flatten()])
                if not hasattr(self._weights[0],'shape'): # no/flat weights
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T)), MT), y[bb_m].flatten())
                else:
                    w = self._weights[0][bb_m].flatten()[:,None]
                    ddx,ddy = np.dot(np.dot(np.linalg.inv(np.dot(MT, MT.T*w)), MT), y[bb_m].flatten())
                if ddx**2 + ddy**2 > self.center_min_dist**2:
                    center = source.center + (ddy, ddx)
                    source.set_center(center)
                    logger.info("Source %d shifted by (%.3f/%.3f) to (%.3f/%.3f)" % (m, ddy, ddx, source.center[0], source.center[1]))

    def _get_shift_differential(self, m):
        # compute (model - dxy*shifted_model)/dxy for first-order derivative
        source = self.sources[m]
        slice_m = source.get_slice_for(self._img.shape)
        k = self.component_of(m, 0)
        model_m = self._models[k][self.sources[m].bb]
        # in self._models, per-source components aren't combined,
        # need to combine here
        for k in range(1,source.K):
            model_m += self._models[k][self.sources[m].bb]

        # get Gamma matrices of source m with additional shift
        offset = source.shift_center
        y_, x_ = source.center_int
        dx = source.center - source.center_int
        pos_x = dx + (0, offset)
        pos_y = dx + (offset, 0)
        dGamma_x = source._gammaOp(pos_x)
        dGamma_y = source._gammaOp(pos_y)
        diff_img = [source.get_model(combine=True, Gamma=dGamma_x), source.get_model(combine=True, Gamma=dGamma_y)]
        diff_img[0] = (model_m-diff_img[0][slice_m])/source.shift_center
        diff_img[1] = (model_m-diff_img[1][slice_m])/source.shift_center
        return diff_img

    """
    def _compute_flux_at_edge(self):
        # compute model flux along the edges
        self.flux_at_edge[0] = model[:,:,-1,:].sum()
        self.flux_at_edge[1] = model[:,:,:,-1].sum()
        self.flux_at_edge[2] = model[:,:,0,:].sum()
        self.flux_at_edge[3] = model[:,:,:0].sum()

    def resize_sources(self):
        for m in range(self.M):
            # TODO: what's the threshold here:
            # I'd say avg flux along edge in band b < avg noise level along edge in b
            at_edge = (self.sources[m].flux_at_edge > model.sum()*flux_thresh) # top, right, bottom, left
            if at_edge.any():
                # TODO: without symmetry constraints, the four edges of the box
                # should be allowed to resize independently
                increase = 10
                self.resize((self.Ny + increase*(at_edge[0] | at_edge[2]), self.Nx + increase*(at_edge[1] | at_edge[3])))
    """
