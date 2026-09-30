import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from optuna import Trial

logger = logging.getLogger(__name__)

########################################################################################
# Parameters. ##########################################################################
########################################################################################

# NB: Make sure to register new parameter types in the _REGISTRY below.

# Base param. ##########################################################################


class Param(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @abstractmethod
    def suggest(self, trial: Trial) -> Any:
        pass

    @classmethod
    @abstractmethod
    def parse(cls, name: str, spec: str) -> "Param":
        """Construct a Param from its DSL string spec."""
        pass


# IntParam. ############################################################################


class IntParam(Param):
    def __init__(self, name: str, low: int, high: int, log: bool = False) -> None:
        super().__init__(name)
        self._low = low
        self._high = high
        self._log = log

    def suggest(self, trial: Trial) -> int:
        param = trial.suggest_int(self._name, self._low, self._high, log=self._log)
        logger.info(
            f"Suggested {self._name}={param} (low={self._low}, high={self._high}, log={self._log})"
        )
        return param

    @classmethod
    def parse(cls, name: str, spec: str) -> "IntParam":
        # tag(log, int(interval(2, 16)))  OR  int(interval(2, 16))
        tags, inner = _unwrap_tags(spec)
        func, args = _parse_call(inner)
        assert func == "int", f"Expected 'int', got '{func}'"
        _, bounds = _parse_call(args[0])  # interval(low, high)
        low, high = int(bounds[0]), int(bounds[1])
        return cls(name, low, high, log="log" in tags)

    def __str__(self) -> str:
        return f"IntParam(name={self._name}, low={self._low}, high={self._high}, log={self._log})"


# FloatParam. ##########################################################################


class FloatParam(Param):
    def __init__(self, name: str, low: float, high: float, log: bool = False) -> None:
        super().__init__(name)
        self._low = low
        self._high = high
        self._log = log

    def suggest(self, trial: Trial) -> float:
        param = trial.suggest_float(self._name, self._low, self._high, log=self._log)
        logger.info(
            f"Suggested {self._name}={param} (low={self._low}, high={self._high}, log={self._log})"
        )
        return param

    @classmethod
    def parse(cls, name: str, spec: str) -> "FloatParam":
        # tag(log, float(interval(0.1, 100.)))  OR  float(interval(0.1, 100.))
        tags, inner = _unwrap_tags(spec)
        func, args = _parse_call(inner)
        assert func == "float", f"Expected 'float', got '{func}'"
        _, bounds = _parse_call(args[0])
        low, high = float(bounds[0]), float(bounds[1])
        return cls(name, low, high, log="log" in tags)

    def __str__(self) -> str:
        return f"FloatParam(name={self._name}, low={self._low}, high={self._high}, log={self._log})"


# CategoricalParam. ####################################################################


class CategoricalParam(Param):
    def __init__(self, name: str, choices: list[Any]) -> None:
        super().__init__(name)
        self._choices = choices

    def suggest(self, trial: Trial) -> Any:
        param = trial.suggest_categorical(self._name, self._choices)
        logger.info(f"Suggested {self._name}={param} (choices={self._choices})")
        return param

    @classmethod
    def parse(cls, name: str, spec: str) -> "CategoricalParam":
        # choice(mean, sum, max)  OR  choice(true, false)
        func, args = _parse_call(spec)
        assert func == "choice", f"Expected 'choice', got '{func}'"
        choices = [_coerce(a) for a in args]
        return cls(name, choices)

    def __str__(self) -> str:
        return f"CategoricalParam(name={self._name}, choices={self._choices})"


# Parse parameters. ####################################################################

_REGISTRY: dict[str, type[Param]] = {
    "int": IntParam,
    "float": FloatParam,
    "choice": CategoricalParam,
}


def parse_param(name: str, spec: str) -> Param:
    """Dispatch a DSL spec string to the correct Param subclass."""
    _, inner = _unwrap_tags(spec)  # strip tag(...) if present
    func, _ = _parse_call(inner)
    if func not in _REGISTRY:
        raise ValueError(f"Unknown param type '{func}' in spec: {spec!r}")
    return _REGISTRY[func].parse(name, spec)


########################################################################################
# Conditional Group. ###################################################################
########################################################################################


@dataclass
class ConditionalGroup:
    conditions: list[tuple[str, Any]]
    params: dict[str, Param]


def parse_conditional_groups(raw: list[dict]) -> list[ConditionalGroup]:
    groups: list[ConditionalGroup] = []
    for entry in raw:
        when = entry["when"]
        # YAML booleans arrive as Python bools, and `str(True)` is "True" while
        # `_coerce` only recognizes the lowercase literal, so stringifying them would
        # produce a condition that can never match a suggested value.
        conditions = [
            (param, value if isinstance(value, bool) else _coerce(str(value)))
            for cond_dict in when
            for param, value in cond_dict.items()
        ]
        params = {
            name: parse_param(name, spec) for name, spec in entry["params"].items()
        }
        groups.append(ConditionalGroup(conditions=conditions, params=params))
    return groups


########################################################################################
# DSL helpers. #########################################################################
########################################################################################


def _unwrap_tags(spec: str) -> tuple[set[str], str]:
    """Peel off tag(...) wrappers, returning (tags, inner_spec)."""
    func, args = _parse_call(spec)
    if func == "tag":
        tags = set(args[:-1])
        return tags, args[-1]
    return set(), spec


def _parse_call(spec: str) -> tuple[str, list[str]]:
    """Parse 'func(arg1, arg2, ...)' → ('func', ['arg1', 'arg2', ...])."""
    match = re.fullmatch(r"(\w+)\((.+)\)", spec.strip(), re.DOTALL)
    if not match:
        return spec.strip(), []
    return match.group(1), _split_args(match.group(2))


def _split_args(s: str) -> list[str]:
    """Split comma-separated args while respecting nested parentheses and brackets."""
    args, depth = [], 0
    buf: list[str] = []
    for ch in s:
        if ch == "," and depth == 0:
            args.append("".join(buf).strip())
            buf = []
        else:
            # Brackets count too, so a list-valued choice such as `[7, 28]` survives as
            # one argument instead of being split on its own comma.
            depth += (ch == "(") - (ch == ")") + (ch == "[") - (ch == "]")
            buf.append(ch)
    if buf:
        args.append("".join(buf).strip())
    return args


def _coerce(value: str) -> Any:
    """Convert YAML-like string literals to Python types."""
    if value == "true":
        return True
    if value == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value  # keep as string (e.g. "mean", "sum", "max")
