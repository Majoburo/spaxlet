import autograd.numpy as np
from autograd.numpy.numpy_boxes import ArrayBox
from autograd.core import VSpace
import logging
import pickle

from .bbox import Box

logger = logging.getLogger("scarlet.component")

class Parameter(np.ndarray):
    def __new__(cls, array, prior=None, step=0, converged=False, fixed=False, **kwargs):
        obj = np.asarray(array, dtype=array.dtype).view(cls)
        obj.prior = prior
        obj.step = 0
        obj.converged = converged
        obj.fixed = fixed
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.prior = getattr(obj, 'prior', None)
        self.step = getattr(obj, 'step_size', 0)
        self.converged = getattr(obj, 'converged', False)
        self.fixed = getattr(obj, 'fixed', False)

    @property
    def _data(self):
        return self.view(np.ndarray)

class SEDParameter(Parameter):
    pass

class MorphParameter(Parameter):
    """
    Specialization of the parameter class to represent a morphology.
    """

    def __new__(cls, array, pixel_center=(0,0), **kwargs):
        obj = super().__new__(cls, array, **kwargs).view(cls)
        obj.pixel_center = pixel_center
        return obj

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        if obj is None: return
        self.pixel_center = getattr(obj, 'pixel_center', (0,0))

    def get_centered_ROI(self, size):
        """
        Extracts a bounding box of desired size centered on the pixel center.

        Parameters
        ----------
        size: `int`
            Size in pixels of the ROI

        Returns
        -------
        bbox: `~scarlet.Box`
            Bounding box centered on the source
        padding:
            Optionally non zero padding needed to get a postage stamp of the
            desired size (in case of border effects).
        """
        radius = size // 2
        left = self.pixel_center[1] - radius
        right = self.pixel_center[1] + radius - 1
        bottom = self.pixel_center[0] - radius
        top = self.pixel_center[0] + radius - 1

        _left = max(left, 0)
        _right = min(right, self.shape[1] -1)
        _bottom = max(bottom, 0)
        _top = min(top, self.shape[0] - 1)

        padding = ((_bottom-bottom, top-_top), (_left-left, right-_right))

        return Box.from_bounds(_bottom, _top, _left, _right), padding


ArrayBox.register(SEDParameter)
ArrayBox.register(MorphParameter)
VSpace.register(SEDParameter, vspace_maker=VSpace.mappings[np.ndarray])
VSpace.register(MorphParameter, vspace_maker=VSpace.mappings[np.ndarray])


class Component():
    """A single component in a blend.

    This class acts as base for building complex :class:`scarlet.source.Source`.

    Parameters
    ----------
    frame: `~scarlet.Frame`
        The spectral and spatial characteristics of this component.
    sed: `~scarlet.Parameter`
        1D array (bands) of the initial SED.
    morph: `~scarlet.Parameter`
        Image (Height, Width) of the initial morphology.
    """

    def __init__(self, frame, sed, morph):
        self._frame = frame

        # set sed and morph
        if isinstance(sed, SEDParameter):
            self._sed = sed
        else:
            self._sed = SEDParameter(sed.copy())
        if isinstance(morph, MorphParameter):
            self._morph = morph
        else:
            self._morph = MorphParameter(morph.copy())

        # Properties used for indexing in the ComponentTree
        self._index = None
        self._parent = None

    @property
    def shape(self):
        """Shape of the image (Channel, Height, Width)
        """
        return self._frame.shape

    @property
    def coord(self):
        """The coordinate in a `~scarlet.component.ComponentTree`.
        """
        if self._index is not None:
            if self._parent._index is not None:
                return tuple(self._parent.coord) + (self._index,)
            else:
                return (self._index,)

    @property
    def frame(self):
        """The frame of this component
        """
        return self._frame

    @property
    def sed(self):
        """Numpy view of the component SED
        """
        return self._sed._data

    @property
    def morph(self):
        """Numpy view of the component morphology
        """
        return self._morph._data

    @property
    def pixel_center(self):
        """ Pixel center of component
        """
        return self._morph.pixel_center

    @property
    def parameters(self):
        return [ p for p in [self._sed, self._morph] if not p.fixed ]

    def get_model(self, *params):
        """Get the model for this component.

        Parameters
        ----------
        params: tuple of optimimzation parameters

        Returns
        -------
        model: array
            (Bands, Height, Width) image of the model
        """
        sed, morph = self.sed, self.morph

        # if params are set they are not Parameters, but autograd ArrayBoxes
        # need to access the wrapped class with _value
        for p in params:
            if isinstance(p._value, SEDParameter):
                sed = p
            if isinstance(p._value, MorphParameter):
                morph = p
        return sed[:, None, None] * morph[None, :, :]

    def get_flux(self):
        """Get flux in every band
        """
        return self.morph.sum() * self.sed

    def update(self):
        """Update the component

        This method can be overwritten in inherited classes to
        run proximal operators or other component update functions
        that will be executed during fitting.
        """
        return self

    def __getstate__(self):
        # needed for pickling to understand what to save
        return tuple([self._sed.copy(), self._morph.copy()])

    def __setstate__(self, state):
        self._sed, self._morph = state

    def save(self, filename):
        fp = open(filename, "wb")
        pickle.dump(self, fp)
        fp.close()

    @classmethod
    def load(cls, filename):
        fp = open(filename, "rb")
        return pickle.load(fp)

class ComponentTree():
    """Base class for hierarchical collections of Components.
    """

    def __init__(self, components):
        """Constructor

        Group a list of `~scarlet.component.Component`s in a hierarchy.

        Parameters
        ----------
        components: list of `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        if not hasattr(components, "__iter__"):
            components = (components,)

        # check type and set coords of subordinate nodes in tree
        self._tree = tuple(components)
        self._index = None
        self._parent = None
        for i, c in enumerate(self._tree):
            if not isinstance(c, ComponentTree) and not isinstance(c, Component):
                raise NotImplementedError("argument needs to be list of Components or ComponentTrees")
            assert c.frame is self.frame, "All components need to share the same Frame"
            c._index = i
            c._parent = self

        self._components = None

    @property
    def components(self):
        """Flattened tuple of all components in the tree.

        CAUTION: Each component in a tree can only be a leaf of a single node.
        While one can construct trees that hold the same component multiple
        times, this method will only return that component at its first
        encountered location
        """
        if self._components is None:
            components = []
            for c in self._tree:
                if isinstance(c, ComponentTree):
                    _c = c.components
                else:
                    _c = [c]
                # check uniqueness
                for __c in _c:
                    if __c not in components:
                        components.append(__c)
            self._components = tuple(components)
        return self._components

    @property
    def n_components(self):
        """Number of components.
        """
        return len(self.components)

    @property
    def K(self):
        """Number of components.
        """
        return self.n_components

    @property
    def frame(self):
        """Frame of the components.
        """
        return self._tree[0].frame

    @property
    def sources(self):
        """Initial list of components or sources that generate the tree.

        This will be different than `self.components` because sources can
        have multiple components.

        Returns
        -------
        The arguments of `__init__`
        """
        return self._tree

    @property
    def n_nodes(self):
        """Number of direct attached nodes.
        """
        return len(self._tree)

    @property
    def coord(self):
        """The coordinate in tree.

        The coordinate can be used to traverse the tree and for `__getitem__`.
        """
        if self._index is not None:
            if self._parent._index is not None:
                return tuple(self._parent.coord) + (self._index,)
            else:
                return (self._index,)

    @property
    def parameters(self):
        pars = []
        for c in self.components:
            pars += c.parameters
        return pars

    def get_model(self, *params):
        """Get the model this component tree

        Parameters
        ----------
        params: tuple of optimization parameters

        Returns
        -------
        model: array
            (Bands, Height, Width) data cube
        """
        model = np.zeros(self.frame.shape)
        if len(params):
            i = 0
            for k,c in enumerate(self.components):
                j = len(c.parameters)
                p = params[i:i+j]
                i += j
                model = model + c.get_model(*p)
        else:
            for c in self.components:
                model = model + c.get_model()

        return model

    def get_flux(self):
        """Get the total flux for all the components in the tree
        """
        for k, component in enumerate(self.components):
            if k == 0:
                model = component.get_flux()
            else:
                model += component.get_flux()
        return model

    def update(self):
        """Update each component

        This method may be overwritten in inherited classes to
        perform updates on multiple components at once
        (for example separating a buldge and disk).
        """
        for component in self.components:
            component.update()

    def __iadd__(self, c):
        """Add another component or tree.

        Parameters
        ----------
        c: `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        c_index = self.n_nodes
        if isinstance(c, ComponentTree):
            self._tree = self._tree + c._tree
        elif isinstance(c, Component):
            self._tree = self._tree + (c,)
        else:
            raise NotImplementedError("argument needs to be Component or ComponentTree")
        c._index = c_index
        c._parent = self
        self._components = None
        return self

    def __getitem__(self, coord):
        """Access node in the tree.

        Parameters
        ----------
        coords: int or tuple of ints

        Returns
        -------
        `~scarlet.component.Component` or `~scarlet.component.ComponentTree`
        """
        if isinstance(coord, (tuple, list)):
            if len(coord) > 1:
                return self._tree[coord[0]].__getitem__(coord[1:])
            else:
                return self._tree[coord[0]]
        elif isinstance(coord, int):
            return self._tree[coord]
        else:
            raise NotImplementedError("coord needs to be index or list of indices")
