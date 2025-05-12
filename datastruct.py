import os
import numpy as np
import json
import gzip
from copy import copy, deepcopy
from pydantic.v1.dataclasses import dataclass, Field
from typing import List, Union, Dict, Any
from datetime import datetime

from bumps.names import Curve, FitProblem
from functions import function_types
from numpy_encoder import NumpyNDArray, numpy_encoder

def find_next_filename(fn: str) -> str:
    """Protects against overwriting files"""

    newfn = fn
    i = 0
    while os.path.exists(newfn):
        i += 1
        pth, ext = os.path.splitext(fn)
        newfn = pth + f'_{i}' + ext

    return newfn

@dataclass
class DataPoint:
    """ Container object for a single data point.

    A "single data point" normally corresponds to a single instrument configuration.

    Required attributes:
    x -- position in instrument configuration space
    y -- measured counting rate
    dy -- uncertainty in measured counting rate (e.g. std dev)
    meastime -- measurement time associated with data point

    Optional attributes:
    movetime (default 0) -- movement time required to set up instrument configuration before measuring
    merit -- value of figure of merit, if applicable, else None

    """

    x: float
    y: float
    dy: float
    meastime: float
    movetime: float = 0
    merit: Union[float, None] = None

    def __repr__(self):

        try:
            reprq = 'x: %0.4f' % self.x
        except TypeError:
            reprq = 'x: ' + ', '.join('{:0.4f}'.format(x) for x in self.x)
        
        return reprq + ('\tTime: %0.1f s' %  self.meastime)

@dataclass(config={'arbitrary_types_allowed': True})
class ExperimentStep:
    """ Container object for a single experiment step.

        Attributes:
        points -- a list of DataPoint objects
        H -- Entropy in all parameters
        H_marg -- Entropy from selected parameters (marginalized entropy)
        H_pars -- list of entropy of inidividual parameters
        foms -- list of the figures of merit for each model
        meastimes -- list of the measurement time proposed for each x value of each model
        movetimes -- list of the movement time proposed for each x value of each model
        yprofs -- list of y(x) profile arrays calculated from each sample from the MCMC posterior
        draw_logp -- 1D vector containing nllf for each sample in draw_pts
        draw_pts -- 2D vector (N_samples x N_parameters) containing posterior PDF samples
        
        TODO: include Settings.sel in each step? Could imagine a case where the selection variables
              change over the course of the experiment

        Methods:
        getdata -- returns all data of type "attr" for data points
        meastime -- returns the total measurement time
        movetime -- returns the total movement time
    """

    points: List[DataPoint]
    H: Union[float, None] = None
    H_marg: Union[float, None] = None
    H_pars: Union[list, None] = None
    foms: Union[NumpyNDArray, None] = None
    meastimes: Union[NumpyNDArray, None] = None
    movetimes: Union[NumpyNDArray, None] = None
    yprofs: Union[NumpyNDArray, None] = None
    draw_logp: Union[NumpyNDArray, None] = None
    draw_pts: Union[NumpyNDArray, None] = None

    def getdata(self, attr):
        # returns all data of type "attr"
        return [getattr(pt, attr) for pt in self.points]

    def meastime(self):

        return sum(pt.meastime for pt in self.points)

    def movetime(self):
 
        return sum(pt.movetime for pt in self.points)
 
@dataclass(config={'arbitrary_types_allowed': True})
class Settings:
    """Experiment settings definition
    
    Attributes:
    ground_truth_pars -- ground truth parameters for simulated experiment. Order not important
    x -- numpy array defining the measurement space
    fit_options -- dictionary of bumps fitter parameters
    entropy_options -- dictionary of entropy calculation options
    sel -- indices of selected parameters of interest. None for all parameters
    thinning -- sparsity of MCMC draw, default 1 (no thinning)
    """

    ground_truth_pars: Dict[str, float]
    x: NumpyNDArray
    fit_options: dict
    entropy_options: dict
    sel: Union[List[int], None] = None
    thinning: int = 1

@dataclass(config={'arbitrary_types_allowed': True})
class Experiment:
    """ Container object for a single experiment step.

        Attributes:
        name -- experiment identifier
        H0 -- initial entropy
        H0_marg -- initial marginalized entropy
        labels -- parameter names from bumps.FitProblem.labels(). Order is important and keys should
                    be the same as Settings.ground_truth_pars
        steps -- a list of ExperimentStep objects
        control -- mark as a control experiment. Each step in ExperimentStep is treated as a single
                    experiment entire, i.e. the total times are not added together

        Methods:
        totaltimes -- returns vector of total time (measure + move) at each step
        meastimes -- returns vector of total measurement time at each step
        movetimes -- returns vector of total movement time at each step
        entropy -- returns vector of total entropy (all parameters) at each step
        entropy_marg -- returns vector of marginalized entropy (parameters of interest) at each step
    """

    name: str
    settings: Settings
    function: function_types
    H0: Union[float, None] = None
    H0_marg: Union[float, None] = None
    labels: Union[List[str], None] = None
    steps: List[ExperimentStep] = Field(default_factory=list)
    control: bool = False
    creation_time: datetime = Field(default_factory=datetime.now)

    def __post_init_post_parse__(self):

        if self.labels is None:
            self.labels = self.make_problem([0], [1], [1]).labels()

    def truey(self):
        """Calculates ground truth y(x) profile"""

        ground_truth_par_list = [self.settings.ground_truth_pars[key] for key in self.labels]

        return self.calc_yprofs(ground_truth_par_list)[0]
    #############NEW FUNCTIONS 
    def get_true_fit_par(self):
        true_par = np.array([self.settings.ground_truth_pars[key] for key, p in self.make_problem([0],[1],             [1]).model_parameters().items() if not p.fixed])
        
        return true_par
        
    def load_pts(self):
        allt = [step.draw_pts for step in self.steps if step.draw_pts is not None]
        return allt
        
    def load_FOM(self):
        allt = [step.foms for step in self.steps if step.draw_pts is not None]
        return allt
        
    
    ###################
    def calc_yprofs(self, pts):
        """Calculates statistical yprofiles from samples of the parameter PDF"""
        yprofs = []
        x = copy(self.settings.x)
        pts = np.array(pts, ndmin=2)

        problem = self.make_problem([0], [1], [1])

        for p in pts:
            problem.setp(p)
            problem.chisq_str()
            yprofs.append(problem.fitness.theory(x))

        return np.array(yprofs)

    def make_problem(self, x, y, dy):
        """Makes a Bumps FitProblem object from "function" definition and data (x, y, dy)"""

        modely = Curve(self.function.func, x, y, dy, name=self.function.name, **self.function.parameters)

        # TODO: might not work for multiple models
        partial = False if len(x) > sum([not p.fixed for p in modely.parameters().values()]) else True

        return FitProblem(modely, partial=partial)

    def totaltimes(self) -> np.ndarray:
        """Returns total time associated with measurement and movement by step
        
            If a control measurement, assumes each measurement and movement was done only once;
                otherwise, use the cumulative sum of all previous steps"""
        
        return self.meastimes() + self.movetimes()

    def meastimes(self) -> np.ndarray:
        """Returns total measurement time by step
        
            If a control measurement, assumes each measurement was done only once;
                otherwise, use the cumulative sum of all previous steps"""
        
        if not self.control:
            allt = np.cumsum([step.meastime() for step in self.steps if step.draw_pts is not None])
        else:
            # assume all movement was done only once
            allt = np.array([step.meastime() for step in self.steps if step.draw_pts is not None])

        return allt

    def movetimes(self) -> np.ndarray:
        """Returns total movement time by step
        
            If a control measurement, assumes each movement was done only once;
                otherwise, use the cumulative sum of all previous steps"""
        
        if not self.control:
            allt = np.cumsum([step.movetime() for step in self.steps if step.draw_pts is not None])
        else:
            # assume all movement was done only once
            allt = np.array([step.movetime() for step in self.steps if step.draw_pts is not None])

        return allt

    def _load_entropy(self, marg=False) -> np.ndarray:
        """Returns entropy associated with measurement and movement by step
        
            Returns total entorpy (all parameters) or marginalized entropy if marg=True"""

        if marg:
            allH = [step.H_marg for step in self.steps if step.draw_pts is not None]
        else:
            allH = [step.H for step in self.steps if step.draw_pts is not None]
        
        return np.array(allH)
    
    def entropy(self) -> np.ndarray:
        """Total entropy associated with all parameters"""

        return self._load_entropy(marg=False)
    
    def entropy_marg(self) -> np.ndarray:
        """Total entropy associated with all parameters"""

        return self._load_entropy(marg=True)
    
    def getdata(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get x, y, dy data from steps"""

        datax = [pt.x for step in self.steps for pt in step.points]
        datay = [pt.y for step in self.steps for pt in step.points]
        datady = [pt.dy for step in self.steps for pt in step.points]

        return np.array(datax), np.array(datay), np.array(datady)
    
    def save(self, fn: Union[str, None] = None, overwrite=False, strip_yprofs=False) -> str:
        """Serializes the Experiment object and saves to a gzipped JSON string
        
        fn: file name string, or None to use object name as file name
        overwrite: if True, overwrites existing files; if False, appends _{i} before
                    the last extension, where i is the number of the next available file name.
        strip_yprofs: if True, strips yprofs from each step before saving. This saves an enormous
            amount of space and time to save

        Returns:
        savefn: saved file name
        """

        # Use object name
        savefn = self.name if fn is None else fn

        # Add '.gz' to the end to show that it is gzipped
        savefn = savefn if savefn.endswith('.gz') else savefn + '.gz'

        # Implement overwrite protection
        savefn = savefn if overwrite else find_next_filename(savefn)

        # Report file name
        print(f'Saving to {savefn}...')

        # strip y profiles for saving
        if strip_yprofs:

            save_self = deepcopy(self)
            for step in save_self.steps:
                step.yprofs = None

        else:

            save_self = self

        # Save experiment to gzipped JSON
        dumpstr = json.dumps(save_self, indent=4, default=numpy_encoder)
        gzip.open(savefn, 'wb').write(bytearray(dumpstr.encode('utf-8')))

        return savefn

    @classmethod
    def load(cls, fn: str, recalc_yprofs: bool = False) -> None:
        """ Loads and deserializes gzipped Experiment object.
                Usage: exp = Experiment.load(fn)
                
            fn: file name of saved Experiment object
            recalc_yprofs: recalculates yprofs for each step if True
        """

        with gzip.open(fn, 'rb') as f:
            newobj = cls(**json.loads(f.read()))

        # recalculate y profiles
        if recalc_yprofs:
            for step in newobj.steps:
                if (step.yprofs is None) & (step.draw_pts is not None):
                    step.yprofs = newobj.calc_yprofs(step.draw_pts)

        return newobj
    
    def load_yprofs(self) -> np.ndarray:
        """Returns all non-None y profiles for all steps"""
        
        all_yprof = [step.yprofs for step in self.steps if step.yprofs is not None]
        
        return np.array(all_yprof)
    
    def load_pts(self) -> np.ndarray:
        """Returns all non-None posterior PDF samples (draw points) for all steps"""

        all_pts = [step.draw_pts for step in self.steps if step.yprofs is not None]
        
        return np.array(all_pts)


