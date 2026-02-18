import object_library 

import particles
import couplings
import lorentz
import parameters
import vertices
import coupling_orders
import write_param_card
import propagators

import function_library

all_particles = particles.all_particles
all_vertices = vertices.all_vertices
all_couplings = couplings.all_couplings
all_lorentz = lorentz.all_lorentz
all_parameters = parameters.all_parameters
all_orders = coupling_orders.all_orders
all_functions = function_library.all_functions
all_propagators = propagators.all_propagators

try:
   import decays
except ImportError:
   pass
else:
   all_decays = decays.all_decays

try:
   import form_factors
except ImportError:
   pass
else:
   all_form_factors = form_factors.all_form_factors

try:
   import CT_vertices
except ImportError:
   pass
else:
   all_CTvertices = CT_vertices.all_CTvertices

try:
   import CT_parameters
except ImportError:
   pass
else:
   all_CTparameters = CT_parameters.all_CTparameters




gauge = [0, 1]

__author__   = "C. Degrande, G. Durieux, F. Maltoni, K. Mimasu, E. Vryonidou, C. Zhang"
__date__     = "2021-05-13"
__version__  = "1.0.3"
__arxiv__      = "[arXiv:2008.11743]"
__url__      = "https://feynrules.irmp.ucl.ac.be/wiki/SMEFTatNLO"

headerinfo = (
  'SMEFT@NLO, v{} from {}'.format(__version__,__date__), 
  __author__, 
  'please refer to {:}'.format(__arxiv__),
  'more information at {}'.format(__url__)
)

maxlen = max([len(x) for x in headerinfo])
hchar = '-'

__header__ = (
'{0}\n'+
'{1} {{:<{2}}} {1}\n'*len(headerinfo)+
'{0}'
).format((maxlen+12)*hchar, 5*hchar, maxlen).format(*headerinfo)




