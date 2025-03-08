"""Gateway notebook for SVG Image Generation"""

import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from kaggle_evaluation.core.base_gateway import GatewayRuntimeError, GatewayRuntimeErrorType, IS_RERUN
import kaggle_evaluation.core.templates
from kaggle_evaluation.svg_constraints import SVGConstraints


class SVGGateway(kaggle_evaluation.core.templates.Gateway):
    def __init__(self, data_path: str | Path | None = None):
        super().__init__(target_column_name='svg')
        self.set_response_timeout_seconds(60 * 5)
        self.row_id_column_name = 'id'
        self.data_path: Path = Path(data_path) if data_path else Path(__file__).parent
        self.constraints: SVGConstraints = SVGConstraints()

    def generate_data_batches(self):
        test = pl.read_csv(self.data_path / 'test.csv')
        for _, group in test.group_by('id'):
            yield group.item(0, 0), group.item(0, 1)  # id, description

    def get_all_predictions(self):
        row_ids, predictions = [], []
        for id, description in self.generate_data_batches():
            svg = self.predict(description)
            self.validate(svg)
            row_ids.append(id)
            predictions.append(svg)

        return predictions, row_ids

    def validate(self, svg: str):
        try:
            self.constraints.validate_svg(svg)
        except ValueError as err:
            msg = f'SVG failed validation: {str(err)}'
            raise GatewayRuntimeError(GatewayRuntimeErrorType.INVALID_SUBMISSION, msg)

    def write_submission(self, predictions: list, row_ids: list) -> Path:
        predictions = pl.DataFrame(
            data={
                self.row_id_column_name: row_ids,
                self.target_column_name: predictions,
            }
        )

        submission_path = Path('/kaggle/working/submission.csv')
        if not IS_RERUN:
            with tempfile.NamedTemporaryFile(prefix='kaggle-evaluation-submission-', suffix='.csv', delete=False, mode='w+') as f:
                submission_path = Path(f.name)

        predictions.write_csv(submission_path)

        return submission_path
