from relicl.model import RelModel
from relicl.typing import FusionType, ModelType

from .extractor import TabFMExtractor, TabICLExtractor, TabPFNExtractor
from .predictor import RelICLPredictor
from .tab_model import TabFMModel, TabICLModel, TabPFNModel


class RelICLModel(RelModel):
    def __init__(self):
        super().__init__()

        # Fine-tuned checkpoints are implemented for TabICL only.
        match self.config.method.model:
            case ModelType.tabicl:
                self._tab_model = TabICLModel()
                self._extractor = TabICLExtractor(self._tab_model)
                self._predictor = RelICLPredictor(self._tab_model)
            case ModelType.tabpfn:
                assert self.config.fusion.type is not FusionType.late, (
                    "late fusion for TabPFN not yet implemented"
                )
                self._tab_model = TabPFNModel()
                self._extractor = TabPFNExtractor(self._tab_model)
                self._predictor = RelICLPredictor(self._tab_model)
            case ModelType.tabfm:
                assert self.config.fusion.type is not FusionType.late, (
                    "late fusion for TabFM not yet implemented"
                )
                self._tab_model = TabFMModel()
                self._extractor = TabFMExtractor(self._tab_model)
                self._predictor = RelICLPredictor(self._tab_model)

    @property
    def tab_model(self) -> TabICLModel | TabPFNModel:
        return self._tab_model

    @property
    def extractor(self) -> TabICLExtractor | TabPFNExtractor:
        return self._extractor

    @property
    def predictor(self) -> RelICLPredictor:
        return self._predictor
