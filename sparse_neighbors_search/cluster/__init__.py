from .minHashSpectralClustering import MinHashSpectralClustering
from .minHashDBSCAN import MinHashDBSCAN
from .minHashClustering import MinHashClustering

import logging
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)