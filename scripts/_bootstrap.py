from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import os
SPLIT_ROOT=Path(os.environ.get('CCC_SPLIT_ROOT',str(ROOT/'local/splits')))
