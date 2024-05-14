from typing import Any
import numpy as np
import ctypes
from scipy.stats import norm, vonmises
from scipy.ndimage import convolve1d
import matplotlib.pyplot as plt
from copy import deepcopy

'''
This module implements the normalization model of attention proposed by Heeger (REF).
It contains 3 classes:
    1. Model is the superordinate class that creates stimulation, excitation, attention, and 
    suppression fields when provided with stimulus mask, attentional bias and model parameters.
        1. stimulus_center = np.array([[x1, y1, feature1], [x2, y2, feature2],...,[xN, yN, featureN]])
            (N x 3) array for N-many stimuli/items, and 3 coordinates (i.e., x, y, feature).
            The spatial coordinates should be in pixels or degrees visual arc.
            The feature coordinates should be in degrees, not radians.
            Each row entry correspond to a stimulus.

        1. attention_center = np.array([[x1, y1, feature1], [x2, y2, feature2],...,[xM, yM, featureM]])
            (M x 3) array for N-many stimuli/items, and 3 coordinates (i.e., x, y, feature)
            Attention centers indicate the biased stimuli. Hence, M <= N.
            It defaults to [None, None, None] where the Attention Field has no bias.
            If the bias is given to only some of the dimensions you should input the unbiased dimension
            with None (e.g., [None, None, -70]: no bias in x & y coordinates but biased at -70 degrees
            in the feature coordinate)

'''

class Model:

    def __init__(self,
                 stimulus_center = np.array([0,0,0]),
                 attention_center = np.array([None, None, None]),
                 impulse_width = [1,1,1],
                 contrast = [1],
                 orientation_span = 360,
                 position_span = 500,             
                 EF_width = [5, 60], # spatial, feature
                 IF_width = [20, 120], # spatial feature
                 AF_amp = [1, 1],
                 AF_base = 1,
                 AF_width = [5, 60],
                 baseline_activity = 5e-7,
                 sigma = 1e-6
                 ):

        # Stimulus field positions 1D or 2D
        self.x = np.arange(-position_span/2, position_span/2)
        self.y = np.arange(-position_span/2, position_span/2)
        self.feature = np.arange(-orientation_span/2, orientation_span/2)
        self.stimulation_center = stimulus_center
        self.stimulation_width = impulse_width
        self.stimulation_amplitude = contrast
        
        self.excitation_width = EF_width
        self.suppression_width = IF_width
        self.attention_center = attention_center
        self.attention_width = AF_width
        self.attention_amplitude = AF_amp
        self.attention_baseline = AF_base
        self.baseline_activity = baseline_activity
        self.sigma = sigma

    def __setattr__(self, name: str, value: Any):

        match name:
            case  "excitation_width" | "stimulation_center" | "stimulation_amplitude" | "stimulation_width" | "suppression_width" | "attention_center" | "attention_width" | "attention_amplitude":
                value = np.array(value)
                if value.ndim == 1: value = np.expand_dims(value,axis=0)
        
        super(type(self),self).__setattr__(name,value)
    
    @property
    def stimulus(self):

        s = Field(
            x = dict(space = self.x,
                     center = self.stimulation_center[:,0],
                     width = self.stimulation_width[:,0],
                     amplitude = self.stimulation_amplitude[:,0],
                     impulse_function = "unit"
                     ),
            y = dict(space = self.y,
                     center = self.stimulation_center[:,1],
                     width = self.stimulation_width[:,1],
                     amplitude = self.stimulation_amplitude[:,0],
                     impulse_function = "unit"
                    ),
            feature = dict(space = self.feature,
                     center = self.stimulation_center[:,2],
                     width = self.stimulation_width[:,2],
                     amplitude = self.stimulation_amplitude[:,0],
                     impulse_function = "unit",
                     isCircular=True)
            )
        return(s)

    @property
    def stimulus_drive(self): 
        
        s = deepcopy(self.stimulus)
        s <<= self._excitation_kernel
        s += self.baseline_activity
        s *= self.attention
        
        return(s)

    @property
    def _excitation_kernel(self):


        e = Field(
            x = Impulse(
                space = self.stimulus.x.space,
                center = [0], 
                width = self.excitation_width[:,0],
                impulse_function = "gaussian",
            ),
            y = Impulse(
                space = self.stimulus.y.space,
                center = [0], 
                width = self.excitation_width[:,0],
                impulse_function = "gaussian",
            ),
            feature = Impulse(
                space = self.stimulus.feature.space,
                center = [0], 
                width = self.excitation_width[:,1],
                impulse_function = "gaussian",
                isCircular=True,
            )
        )

        return(e)
    
    @property
    def attention(self):

        a = Field(
            x = Impulse(
                space = self.stimulus.x.space,
                center = self.attention_center[:,0], 
                width = self.attention_width[:,0],
                amplitude = self.attention_amplitude[:,0],
                impulse_function = "gaussian"
            ),
            y = Impulse(
                space = self.stimulus.y.space,
                center = self.attention_center[:,1], 
                width = self.attention_width[:,0],
                amplitude = self.attention_amplitude[:,0],
                impulse_function = "gaussian"
            ),
            feature = Impulse(
                space = self.stimulus.feature.space,
                center = self.attention_center[:,2], 
                width = self.attention_width[:,1],
                amplitude = self.attention_amplitude[:,1],
                impulse_function = "gaussian",
                isCircular=True
            )
        ) + self.attention_baseline

        return(a)
    
    @property
    def suppressive_drive(self): 
        
        s = deepcopy(self.stimulus_drive)
        s <<= self._suppression_kernel
        s += self.sigma

        return(s)

    @property
    def _suppression_kernel(self):

        s = Field(
            x = Impulse(
                space = self.stimulus.x.space,
                center = [0], 
                width = self.suppression_width[:,0],
                impulse_function = "gaussian"
            ),
            y = Impulse(
                space = self.stimulus.y.space,
                center = [0], 
                width = self.suppression_width[:,0],
                impulse_function = "gaussian"
            ),
            feature = Impulse(
                space = self.stimulus.feature.space,
                center = [0], 
                width = self.suppression_width[:,1],
                impulse_function = "gaussian",
                isCircular=True
            )
        )

        return(s)
    
    @property
    def response(self):

        r = deepcopy(self.stimulus_drive)
        r /= self.suppressive_drive

        return(r)
    
class Field:

    def __init__(self,**kwargs):
        
        for keyN, valN in kwargs.items():

            match keyN:

                case "x" | "y" | "feature": 

                    if isinstance(valN, Impulse):
                        super(type(self),self).__setattr__(keyN,valN)
                    else:
                        super(type(self),self).__setattr__(keyN, Impulse(**valN))

        self.__make_field__()

    @staticmethod
    def __return_numeric__(x):
        
        if isinstance(x, Field): return(x.field)
        elif isinstance(x, Impulse): return(x.impulse)
        elif isinstance(x, float | int): return(x)
        else:
            TypeError("The input must be a Field, an Impulse, a scalar, or an np.array.")

    @staticmethod
    def __return_field__(x):

        if isinstance(x, Field): return(x)
        elif isinstance(x, float | int): return(x)
        else: 
            raise TypeError("The operation could be implemented between variables of Field type only.")
    
    @staticmethod
    def __return_kernel__(x):
        if isinstance(x, Field): 

            return([getattr(x, dimN) for dimN in x.dimensions])
        
        else:
            TypeError("The input is not of Impulse type.")
        
    @staticmethod
    def __convolve__(A, kernels, isCircular):
        """
        Convolves a 3D NumPy array with a list of 1D kernels using scipy.ndimage.convolve1d.

        Args:
            A (np.ndarray): The 3D input array.
            kernels (list): A list of three 1D NumPy arrays representing the kernels.
            isCircular (list): A list of three booleans indicating circular convolution.

        Returns:
            np.ndarray: The convolved 3D array.
        """

        if len(kernels) != 3 or len(isCircular) != 3:
            raise ValueError("kernels and isCircular must have length 3")

        result = A.copy() 
        for dim in range(3):
            if isCircular[dim]:
                mode = 'wrap'  # Circular convolution mode in scipy
            else:
                mode = 'constant'  # Use constant padding for non-circular 

            result = convolve1d(result, kernels[dim], axis=dim, mode=mode)

        return result

    @staticmethod
    def convolve(A, kernels):
        """
        Convolve each dimension of the N-dimensional matrix A with the corresponding vector in kernels.
        
        Parameters:
        A : Field
            A.field is an N-dimensional matrix.
        kernels : list of Impulse
            List of kernels of class Impulse to be used as kernels for convolution.
           
        Returns:
        result : numpy.ndarray
            Convolved N-dimensional matrix.
        """
        isCircular, kernels_mat = [], []
        for whKern, kernN in enumerate(kernels):
            kernels_mat.append(kernN.impulse.squeeze())
            isCircular.append(kernN._isCircularSpace)
        
        return Field.__convolve__(A.field, kernels_mat, isCircular)

        
            
    def __lshift__(self,other): 
        
        op = deepcopy(self) 
        op <<= other

        return(op)
    
    def __ilshift__(self,other):

        other = Field.__return_kernel__(other)
        self.field = Field.convolve(self, other)

        return(self)
    
    def __mul__(self, other):

        op = deepcopy(self) 
        op *= other

        return(op)
        
    def __imul__(self,other):

        other = Field.__return_numeric__(other)
        self.field = self.field * other

        return(self)
    
    def __truediv__(self, other): # elementwise division
        
        op = deepcopy(self) 
        op /= other

        return(op)  
      
    def __itruediv__(self,other): 

        other = Field.__return_numeric__(other)
        self.field = self.field / other

        return(self)
    
    def __add__(self, other): # elementwise division

        op = deepcopy(self) 
        op += other

        return(op)     
    
    def __iadd__(self,other):

        other = Field.__return_numeric__(other)
        self.field = self.field + other

        return(self)
    
    def __sub__(self, other): # elementwise division

        op = deepcopy(self) 
        op -= other

        return(op) 
        
    def __isub__(self,other): 

        other = Field.__return_numeric__(other)
        self.field = self.field + other

        return(self)

    def plot(self, **kwargs):

        xlabels = ["X Position", "Y Position", "Feature"]
        letters = np.array(list(map(chr,range(ord('A'), ord('Z')+1))))
        idx = np.zeros([self.n_dimensions, self.n_dimensions+1])
        idx[:,0] = range(self.n_dimensions)
        idx[:,1:] = np.tile(self.n_dimensions+np.arange(self.n_dimensions),[self.n_dimensions,1])
        subplot_mosaic = '\n'.join([''.join(x) for x in np.take(letters, idx.astype(int))])

        # fig, ax = plt.subplots(self.n_dimensions, self.n_dimensions + 1,layout="constrained")
        ax = plt.figure(layout="constrained", figsize = (5*self.n_dimensions+1,5)).subplot_mosaic(subplot_mosaic)

        for whDim, dimN in enumerate(letters[0:self.n_dimensions]):
            # ax[dimN].plot(getattr(self, self.dimensions[whDim]).space.T,)
            ax[dimN].plot(getattr(self, self.dimensions[whDim]).space.T, getattr(self, self.dimensions[whDim]).impulse.T)
            ax[dimN].set_xlabel(xlabels[whDim])

        for whDim, dimN in enumerate(letters[self.n_dimensions:(2*self.n_dimensions)]):
            imN = self.field.mean(axis=whDim)
            ax[dimN].imshow(imN,cmap="gray")
            # labelN = np.delete(xlabels,whDim)
            # ax[dimN].set_xlabel(labelN[0])
            # ax[dimN].set_ylabel(labelN[1])
        return(ax)
    
    def __make_field__(self):
        self._field = np.expand_dims(self.position, axis=-1) @ self.feature.impulse
        return(self)
    
    @property
    def field(self):return(self._field)
    
    @field.setter
    def field(self, val): self._field = val
    
    @property
    def position(self):

        if self.n_spatial_dimensions == 1: return(self.x)
        elif self.n_spatial_dimensions == 2: return(self.y @ self.x)

    @property
    def dimensions(self): return(
        [d for d in ["x", "y", "feature"] if hasattr(self, d)]
    )

    @property
    def spatial_dimensions(self): return(
        [d for d in ["x", "y"] if hasattr(self, d)]
    )
    
    @property
    def n_dimensions(self): return(len(self.dimensions))

    @property
    def n_spatial_dimensions(self): 
        return(len(self.spatial_dimensions))
    
    @property
    def n_centers(self): return(self._default_dimension.n_centers)

    @property
    def _default_dimension(self):
        if hasattr(self, "x"): return(self.x)
        elif hasattr(self,"y"): return(self.y)
        else: return(self.feature)

    

class Impulse:

    def __init__(self, 
                 space = np.arange(-180,180),
                 center = [0], 
                 width = 5,
                 amplitude = 1, 
                 baseline = 5e-7,
                 impulse_function = "gaussian",
                 isCircular = False):
        
        self._isCircularSpace = isCircular
        self.space = np.array(space)
        self._kernel = None
        self._scalar = 1
        self._divider = 1
        self.impulse_function = impulse_function
        self.center = np.array(center)
        self.width = np.array(width)
        self.amplitude = np.array(amplitude) 
        self.baseline = baseline
        self.__make_impulse__()               
        

    def __setattr__(self, name: str, value: Any):

        match name:
            case "impulse_function":

                if isinstance(value,str):
                    match value:
                        case "gauss" | "gaussian" | "norm" | "normal":

                            if self._isCircularSpace:
                                value = lambda cent,width,amp: amp*vonmises.pdf(self.space * np.pi /180, kappa = width * np.pi /180, loc = cent * np.pi /180)
                            else:
                                value = lambda cent,width,amp: amp*norm.pdf(self.space, cent, width)
                            super(type(self),self).__setattr__("_scalar_function", lambda x: self.__scale_sum_to_one(x))

                        case "point" | "unit":

                            value = lambda cent, width=1,amp=1: amp*self._unit_impulse(
                                self.n_points, 
                                np.all([self.space >= (cent - width/2), self.space <= (cent +  width/2)], axis=0)
                                )
                            super(type(self),self).__setattr__("_scalar_function", lambda x: self.__scale_to_range(x))
                            
            case "width" | "amplitude":

                value = np.array(value)
                if value.size == 1 and self.n_centers > 1:
                    value = np.tile(value, self.n_centers)

                if value.ndim == 1: value = np.expand_dims(value,axis=0)
            
            case "center":

                value = np.array(value)               

                if np.any(value == None):
                    
                    super(type(self),self).__setattr__("impulse_function", lambda *args: np.ones(self.space.shape))
                    super(type(self),self).__setattr__("_scalar_function", lambda x: x)
                
                elif value.ndim == 1:
                    
                    value = np.expand_dims(value,axis=0)

            case "space": 

                if value.ndim == 1: value = np.expand_dims(value,axis=0)
     
        super(type(self),self).__setattr__(name, value)

    def plot(self, **kwargs):
        return(plt.plot(self.space, self.impulse, **kwargs))
    
    # Overriding basic operators
    
    def __lshift__(self, other): # new syntax for convolution <<

        op = deepcopy(self)
        op <<= other
        return(op)
    
    def __ilshift__(self, other):

        self.impulse = self.convolve(self.impulse, self.__return_impulse__(other), isCircular=self._isCircularSpace)
        return(self)
    
    def __matmul__(self, other): # matrix multiplication @
        return(self.impulse.T @ self.__return_impulse__(other))

    def __mul__(self, other): # elementwise multiplication

        op = deepcopy(self)
        op *= other
        return(op)
    
    def __imul__(self, other):    
        self.impulse = self.impulse * self.__return_impulse__(other)
        return(self)
    
    def __truediv__(self, other): # elementwise division
        op = deepcopy(self)
        op *= other
        return(op)
    
    def __itruediv__(self, other):
        
        self.impulse = self.impulse / self.__return_impulse__(other)
        return(self)
    
    def __add__(self,other):
        op = deepcopy(self)
        op *= other
        return(op)
    
    def __iadd__(self, other):
        
        self.impulse = self.impulse + self.__return_impulse__(other)
        return(self)
    
    def __sub__(self, other):
        return(self.impulse - self.__return_impulse__(other))
    
    def __isub__(self, other):
        
        self.impulse = self - other
        return(self)

    def __make_impulse__(self):

        if self.n_centers==1:
            k = self.impulse_function(self.center, self.width, self.amplitude)
        else:
            k = self._scalar_function(
                np.vstack([
                    self.impulse_function(centN, widthN, ampN)
                    for centN, widthN, ampN in zip(self.center.T, self.width.T, self.amplitude.T)
                ]).sum(axis=0))
        if k.ndim == 1: k = np.expand_dims(k,axis=0)

        self.impulse = k + self.baseline
        
        return(self)
    
    @property
    def impulse(self): return(self._impulse)

    @impulse.setter
    def impulse(self, value): self._impulse = value       
    
    # Dependent Properties
    @property
    def n_centers(self): return(self.center.size)
    @property
    def n_points(self): return(self.space.size)

    # Static Methods
    @staticmethod
    def __scale_sum_to_one(x): 
        x = np.array(x)
        return(x/x.sum())
    
    @staticmethod
    def __scale_to_range(x, range = [0,1]):
        x = np.array(x)
        x -= x.min()
        x = x/x.max()
        return(x)       
    
    @staticmethod
    def _unit_impulse(size,cent_idx):
        x = np.zeros(size)
        x[np.squeeze(cent_idx)] = 1
        return(x)
    
    # @staticmethod
    # def convolve(kern1, kern2, isCircular = False):

    #     # if one of the kernels is actually None, returns the non-None type kernel
    #     if kern1 is None: return(kern2)
    #     elif kern2 is None: return(kern1)

    #     if isCircular: 

    #         return(Impulse.circconv(kern1, kern2))

    #     else:

    #         return(linconv(kern1, kern2, mode = "same"))
    
    # @staticmethod
    # def circconv(signal, kern):
    #     return np.real(np.fft.ifft( np.fft.fft(signal)*np.fft.fft(kern) ))
    
    @staticmethod
    def __return_impulse__(x):
        
        if isinstance(x, Impulse): return(x.impulse)
        else: return(x)
    
        





