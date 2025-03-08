import inspect
import pathlib
from types import ModuleType

from kaggle_evaluation.core import relay, templates
from kaggle_evaluation.svg_gateway import SVGGateway


def test(model_cls: type, data_path: str | pathlib.Path | None = None) -> None:
    '''Tests this competition's inference loop over the given Model class.
    
    The provided Model class should have a `predict` function which accepts input(s)
    and returns output(s) with the shapes and types required by this competition.
    This function performs best-effort validation of this by running an inference
    loop with a dummy test set over Model.predict.
    By default the test set is taken from the `kaggle_evaluation` directory, but you
    may override to another directory with the same test file structure via the
    `data_path` arg.'''
    print('Creating Model instance...')
    model = model_cls()
    if not hasattr(model, 'predict') or not inspect.ismethod(model.predict):
        msg = f'Model does not have method predict.'
        raise ValueError(msg)

    print('Running inference tests...')
    server = relay.define_server(model.predict)
    server.start()
    try:
        gateway = SVGGateway(data_path)
        submission_path = gateway.run()
        print(f'Wrote test submission file to "{str(submission_path)}".')
    except Exception as err:
        raise err from None
    finally:
        server.stop(0)

    print('Success!')


def _run_gateway() -> None:
    '''Internal function for running the Gateway during a Kaggle scoring session.
    
    Starts a scoring session which assumes existence of an Inference Server to return
    inferences over the test set.'''
    gateway = SVGGateway()
    gateway.run()


def _run_inference_server(module: ModuleType) -> None:
    '''Internal function for running the Inference Server during a Kaggle scoring session.
    
    Takes the user's submitted, imported module and sets up the inference server exposing
    their required method(s).'''
    model = module.Model()
    server = templates.InferenceServer(model.predict)
    server.serve()