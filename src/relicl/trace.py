import logging
import re
import time
from collections import OrderedDict
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from hydra.core.hydra_config import HydraConfig

import relicl.typing
from relicl.config import RelICLConfig, WithConfig

logger = logging.getLogger(__name__)

########################################################################################
# Representers for various types in YAML

yaml.add_representer(
    np.datetime64, lambda dumper, data: dumper.represent_str(data.__str__())
)

yaml.add_representer(
    np.float16, lambda dumper, data: dumper.represent_float(data.item())
)
yaml.add_representer(
    np.float32, lambda dumper, data: dumper.represent_float(data.item())
)
yaml.add_representer(
    np.float64, lambda dumper, data: dumper.represent_float(data.item())
)

yaml.add_representer(np.int16, lambda dumper, data: dumper.represent_int(data.item()))
yaml.add_representer(np.int32, lambda dumper, data: dumper.represent_int(data.item()))
yaml.add_representer(np.int64, lambda dumper, data: dumper.represent_int(data.item()))

yaml.add_representer(
    np.ndarray, lambda dumper, data: dumper.represent_list(data.tolist())
)

yaml.add_representer(
    OrderedDict, lambda dumper, data: dumper.represent_ordered_dict(data)
)

yaml.add_representer(tuple, lambda dumper, data: dumper.represent_list(data))

yaml.add_representer(
    pd.Timestamp,
    lambda dumper, data: dumper.represent_scalar(
        "tag:yaml.org,2002:str", data.isoformat()
    ),
)


# Write every enum as its plain name.
for _T in vars(relicl.typing).values():
    if isinstance(_T, type) and issubclass(_T, StrEnum) and _T is not StrEnum:
        yaml.add_representer(
            _T,
            lambda dumper, data: dumper.represent_scalar(
                "tag:yaml.org,2002:str", data.name
            ),
        )

########################################################################################
# Tracing code


class Trace(WithConfig):
    _instance = None

    def __init__(self):
        raise RuntimeError("Call instance() instead.")

    @classmethod
    def reset(cls) -> None:
        cls._instance = None

    @classmethod
    def instance(cls) -> "Trace":
        if cls._instance is None:
            config = RelICLConfig.instance()
            cls._instance = cls.__new__(cls)
            cls.config = config
            cls._instance.name = HydraConfig.get().job.name  # type: ignore[attr-defined]
            cls._instance.id = HydraConfig.get().job.id  # type: ignore[attr-defined]
            cls._instance.filename = Path(  # type: ignore[attr-defined]
                HydraConfig.get().runtime.output_dir + "/trace.yaml"
            )
            cls._instance.disabled_events_pattern = re.compile(  # type: ignore[attr-defined]
                config.output.trace.disabled_events_regex
            )
            cls._instance.enabled_events_pattern = re.compile(
                config.output.trace.enabled_events_regex
            )  # type: ignore[attr-defined]
            if config.output.trace.enable:
                logger.info(f"Tracing to {cls._instance.filename}.")  # type: ignore[attr-defined]
            else:
                logger.info(f"Tracing disabled.")

        return cls._instance

    def is_traced(self, event):
        return self.config.output.trace.enable and (
            not self.disabled_events_pattern.fullmatch(event)
            or self.enabled_events_pattern.fullmatch(event)
        )

    def trace(self, event, filename=None, echo=None, **kwargs):
        if not self.is_traced(event):
            return

        if echo is None:
            echo = self.config.output.trace.echo

        if filename is None:
            filename = self.filename
        elif "/" not in filename:
            filename = HydraConfig.get().runtime.output_dir + "/" + filename

        Trace.write(event, filename, name=self.name, id=self.id, echo=echo, **kwargs)

    @staticmethod
    def write(event, filename, name, id, echo=None, **kwargs):
        entry = OrderedDict()
        entry["name"] = name
        entry["id"] = id
        entry["time_ns"] = time.time_ns()
        entry["event"] = event
        entry.update(kwargs)
        with open(filename, "a+") as f:
            line = yaml.dump(entry, width=float("inf"), default_flow_style=True).strip()
            f.write(line)
            f.write("\n")
            if echo:
                logger.info(line)


def load_trace(filename: str) -> list[Any]:
    result = []
    for line in open(filename, "rt"):
        record = yaml.load(line, Loader=yaml.SafeLoader)
        result.append(record)
    return result
