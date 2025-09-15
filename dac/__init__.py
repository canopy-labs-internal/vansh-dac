__version__ = "1.0.0"

# preserved here for legacy reasons
__model_version__ = "latest"

import audiotools

audiotools.ml.BaseModel.INTERN += ["dac.**"]
audiotools.ml.BaseModel.EXTERN += ["einops"]


from . import nn
from . import model
from . import utils
from .model.dac import DAC
from .model.snac import SNAC
from .model import DACFile