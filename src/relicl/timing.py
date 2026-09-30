import functools
import inspect
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import ParamSpec, TypeVar

from relicl.config import RelICLConfig
from relicl.trace import Trace
from relicl.typing import TimingEvent

P = ParamSpec("P")
R = TypeVar("R")


@contextmanager
def timed_step(
    event: str, event_type: TimingEvent = TimingEvent.any, **kwargs: object
) -> Iterator[None]:
    """Context manager that measures and logs the duration of a code block.

    Measures the execution time of the code within the context and traces it
    with high precision using perf_counter_ns. Both the duration in nanoseconds
    and seconds are recorded (the latter for human readability).

    Args:
        event: Name of the event/step being timed. Used as part of the trace
            event key (prefixed with "DURATION-").
        **kwargs: Additional keyword arguments to include in the trace data.
    """

    if RelICLConfig.has_instance():
        trace = Trace.instance()
        start_ns = time.perf_counter_ns()

        try:
            yield
        finally:
            config = RelICLConfig.instance()
            if config.output.trace.timings:
                duration_ns = time.perf_counter_ns() - start_ns
                duration_s = duration_ns / 1_000_000_000
                trace.trace(
                    f"DURATION-{event}",
                    timing_event=event_type,
                    duration_ns=duration_ns,
                    duration_s=duration_s,
                    **kwargs,
                )
    else:
        # do nothing if we do not have RelICL running propery
        yield


def timed(
    event: str | None = None,
    event_type: TimingEvent = TimingEvent.any,
    **trace_kwargs: object,
):
    """Decorator that measures and logs the execution time of a function.

    Wraps a function to automatically time its execution and record the duration
    in both nanoseconds and seconds. If no event name is provided, the decorated
    function's name is used.

    Args:
        event: Name of the event/step being timed. If None, the decorated
            function's __name__ is used. Used as part of the trace event key
            (prefixed with "DURATION-").
        **trace_kwargs: Additional keyword arguments to include in the trace data.

    Returns:
        A decorator that wraps the target function with timing functionality.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        params = list(inspect.signature(fn).parameters)
        first_param = params[0] if params else None
        is_classmethod = first_param == "cls"
        is_instance_method = first_param == "self"

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            event_name = event
            if event_name is None:
                if is_classmethod and args:
                    event_name = f"{args[0].__name__}.{fn.__name__}"  # type: ignore[attr-defined]
                elif is_instance_method and args:
                    event_name = f"{args[0].__class__.__name__}.{fn.__name__}"
                else:
                    event_name = fn.__name__
            with timed_step(event_name, event_type, **trace_kwargs):
                return fn(*args, **kwargs)

        return wrapper

    return decorator
