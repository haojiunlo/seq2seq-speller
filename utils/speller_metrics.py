"""Speller metrics"""

import datasets
import numpy as np


_CITATION = """\
@InProceedings{huggingface:metric,
title = {Speller metric},
authors={orca},
year={2022}
}
"""

_DESCRIPTION = """\
This new metric is designed to evaluate speller performance
"""


_KWARGS_DESCRIPTION = """
Calculates how good are predictions given some references, using certain scores
Args:
    predictions: list of predictions to score. Each predictions
        should be a string with tokens separated by spaces.
    references: list of reference for each prediction. Each
        reference should be a string with tokens separated by spaces.
Returns:
    accuracy: description of the first score,
    another_score: description of the second score,
Examples:
    Examples should be written in doctest format, and should illustrate how
    to use the function.
    >>> my_new_metric = datasets.load_metric("my_new_metric")
    >>> results = my_new_metric.compute(references=[0, 1], predictions=[0, 1])
    >>> print(results)
    {'accuracy': 1.0}
"""


@datasets.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class SpellerMetric(datasets.Metric):
    """TODO: Short description of my metric."""

    def _info(self):
        # TODO: Specifies the datasets.MetricInfo object
        return datasets.MetricInfo(
            # This is the description that will appear on the metrics page.
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            # This defines the format of each prediction and reference
            features=datasets.Features({
                'predictions': datasets.Value('string'),
                'references': datasets.Value('string'),
            }),
            # Homepage of the metric for documentation
            homepage="http://metric.homepage",
            # Additional links to the codebase or references
            codebase_urls=["http://github.com/path/to/codebase/of/new_metric"],
            reference_urls=["http://path.to.reference.url/new_metric"]
        )

    def _download_and_prepare(self, dl_manager):
        """Optional: download external resources useful to compute the scores"""
        pass

    def _compute(self, predictions, references, inputs):
        """Returns the scores"""
        return _metrics(predictions, references, inputs)
    
    
def _metrics(predictions, references, inputs):
    predictions, references, inputs = np.array(predictions), np.array(references), np.array(inputs)
    pass_fail = predictions == references
    acc = pass_fail.mean()
    pre = pass_fail[predictions != inputs].mean()
    rec = pass_fail[references != inputs].mean()
    far = (~pass_fail[references == inputs]).mean()
    return {"acc": acc, "pre": pre, "rec": rec, "far": far}
