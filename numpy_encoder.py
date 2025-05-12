import numpy as np

from typing import Any
from pydantic.v1.json import custom_pydantic_encoder

class NumpyNDArray(np.ndarray):
    """Class for custom pydantic validation. Used for 
        deserializing saved Numpy ndarrays"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
    @classmethod
    def validate(cls, v: Any) -> np.ndarray:
        # validate data...
        return np.array(v)

# Custom encoder function for serializing data structures with numpy ndarray support
numpy_encoder = lambda obj: custom_pydantic_encoder({
                    np.ndarray: lambda o: o.tolist()
               }, obj)