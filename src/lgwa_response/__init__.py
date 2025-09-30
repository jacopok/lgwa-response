from pathlib import Path
import os

# hacky way to get the path of this module
from . import lgwa_settings

data_path = Path(os.path.abspath(lgwa_settings.__file__)).resolve().parent / "data"
