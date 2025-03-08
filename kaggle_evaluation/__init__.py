'''
Competition helper module for the Drawing With LLMs Kaggle Competition.

Supports the `test` function for users to validate their Model, and internal functions
used by Kaggle's scoring system.
'''

import pathlib
import sys

# Provide additional import management since grpc_tools.protoc doesn't support relative imports
module_path = pathlib.Path(__file__).parent
gen_path = module_path / 'core' / 'generated'

if not (gen_path / 'kaggle_evaluation_pb2.py').exists():
    msg = 'Missing required kaggle_evaluation proto / gRPC generated files.'
    raise ImportError(msg)

sys.path.append(str(module_path))
sys.path.append(str(gen_path))

from .svg import test, _run_gateway, _run_inference_server
__all__ = ['test']

__version__ = '0.5.0'
