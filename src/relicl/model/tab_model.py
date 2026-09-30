from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin, RegressorMixin

from relicl.config import WithConfig


class TabModel(ABC, WithConfig):
    """The underlying tabular model"""

    @abstractmethod
    def new_classification_model(self, finetuned: bool = False) -> ClassifierMixin:
        """Creates a new classification instance of the tabular model.

        Only callers that may use a fine-tuned checkpoint pass `finetuned=True`.
        """

    @abstractmethod
    def new_regression_model(self, finetuned: bool = False) -> RegressorMixin:
        """Creates a new regression instance of the tabular model.

        Only callers that may use a fine-tuned checkpoint pass `finetuned=True`.
        """
