from pathlib import Path 
import os
from monitoring.monitor import return_mapping

home = Path.home()
cwd = Path.cwd()

mapping = return_mapping()