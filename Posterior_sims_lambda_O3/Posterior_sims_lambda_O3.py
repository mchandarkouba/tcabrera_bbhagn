"""! Script for the calculations described in https://arxiv.org/abs/2103.16069 """

###############################################################################

import sys

# Local imports
sys.path.append(pa.dirname(pa.dirname(__file__)))
from utils.simulate import main as sim_main

###############################################################################

config_path = sys.argv[1]
sim_main(config_path)
