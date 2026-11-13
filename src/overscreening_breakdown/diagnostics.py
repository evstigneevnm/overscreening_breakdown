"""Solver-scoped progress events and native-precision streaming diagnostics.

Reporting never evaluates a numerical callback. Events contain scalar metadata,
not coefficient vectors, matrices, problem objects, or arithmetic contexts.
"""

from contextlib import contextmanager
from dataclasses import dataclass, field
from decimal import Decimal
import math
from numbers import Integral, Real
import sys
from time import perf_counter
from types import MappingProxyType
from typing import Mapping

import numpy as np

from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


class ProgressCallbackError(RuntimeError):
    """An observer failed; its original exception is retained as the cause.

    A distinct type prevents an observer's OverflowError or LinearSolveError
    from being mistaken for a recoverable numerical failure.
    """


def _snapshot(value, arithmetic=None):
    if hasattr(value, "_mpf_"):
        # Legacy collocation supplies global-context mpf instances. Retain
        # their value in the solver's owner so arithmetic on collected events
        # stays at the requested precision after global precision is restored.
        if arithmetic is not None and arithmetic.use_mpmath:
            return arithmetic.scalar(value)
        return value
    if value is None or isinstance(value, (str, bool, Integral, Real, Decimal, np.bool_)):
        return value
    if isinstance(value, tuple):
        return tuple(_snapshot(item, arithmetic) for item in value)
    raise TypeError("progress metadata must contain immutable real scalars or tuples")


def _mapping(values, arithmetic=None):
    if not isinstance(values, Mapping):
        raise TypeError("progress data must be a mapping")
    if any(not isinstance(name, str) for name in values):
        raise TypeError("progress metadata keys must be strings")
    return MappingProxyType({name: _snapshot(value, arithmetic) for name, value in values.items()})


@dataclass(frozen=True)
class ProgressEvent:
    """One immutable observation; ``data`` retains native numerical values."""

    kind: str
    level: int
    solver: str
    method: str
    elapsed: float
    data: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("progress event kind must be a nonempty string")
        if (isinstance(self.level, (bool, np.bool_)) or not isinstance(self.level, Integral)
                or not 1 <= self.level <= 3):
            raise ValueError("event level must be an integer from 1 through 3")
        if not isinstance(self.solver, str) or not isinstance(self.method, str):
            raise TypeError("progress solver and method labels must be strings")
        if (isinstance(self.elapsed, (bool, np.bool_)) or not isinstance(self.elapsed, Real)
                or not math.isfinite(self.elapsed) or self.elapsed < 0):
            raise ValueError("elapsed time must be finite and nonnegative")
        object.__setattr__(self, "data", _mapping(self.data))


@dataclass
class _SharedState:
    callback: object
    stream: object
    clock: object
    started: float
    last_output: dict = field(default_factory=dict)
    callback_failed: bool = False


class ProgressReporter:
    """Dispatch observations and print verbosity-filtered, flushed text.

    The optional observer receives every emitted event independently of text
    verbosity and throttling. Frequent evaluation/trial text can be throttled;
    phase transitions and final outcomes are always unthrottled. No event
    history is stored. ``bind`` adds immutable context without modifying its
    parent and retains the same clock, observer, stream, and throttle state.
    """

    def __init__(self, verbosity=0, progress_callback=None, arithmetic=None,
                 solver="", method="", progress_interval=1.0, stream=None,
                 clock=perf_counter):
        if (isinstance(verbosity, (bool, np.bool_)) or not isinstance(verbosity, Integral)
                or not 0 <= verbosity <= 3):
            raise ValueError("verbosity must be an integer from 0 through 3")
        if progress_callback is not None and not callable(progress_callback):
            raise TypeError("progress_callback must be callable or None")
        if (isinstance(progress_interval, (bool, np.bool_))
                or not isinstance(progress_interval, Real)
                or not math.isfinite(progress_interval) or progress_interval < 0):
            raise ValueError("progress_interval must be finite and nonnegative")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if stream is not None and not (callable(getattr(stream, "write", None))
                                       and callable(getattr(stream, "flush", None))):
            raise TypeError("stream must provide write() and flush()")
        self.verbosity = int(verbosity)
        self.progress_interval = float(progress_interval)
        self.arithmetic = arithmetic
        self.solver, self.method = str(solver), str(method)
        self._context = MappingProxyType({})
        self._shared = _SharedState(progress_callback, stream, clock, clock())

    @property
    def arithmetic(self):
        return self._arithmetic

    @arithmetic.setter
    def arithmetic(self, owner):
        self._arithmetic = owner
        self._precision_signature = None if owner is None else arithmetic_signature(owner)

    def check_precision(self):
        if self.arithmetic is not None:
            require_precision(self.arithmetic, self._precision_signature,
                              subject="progress reporter", rebuild="start a new solve or reporter")

    @property
    def enabled(self):
        return bool(self.verbosity or (self._shared.callback is not None
                                      and not self._shared.callback_failed))

    def bind(self, **context):
        self.check_precision()
        result = object.__new__(type(self))
        result.verbosity = self.verbosity
        result.progress_interval = self.progress_interval
        result.arithmetic = self.arithmetic
        result.solver = str(context.pop("solver", self.solver))
        result.method = str(context.pop("method", self.method))
        result._context = _mapping({**self._context, **context}, self.arithmetic)
        result._shared = self._shared
        return result

    @staticmethod
    def _level(level):
        if (isinstance(level, (bool, np.bool_)) or not isinstance(level, Integral)
                or not 1 <= level <= 3):
            raise ValueError("event level must be an integer from 1 through 3")

    @staticmethod
    def _throttled(kind, throttle):
        return bool(throttle and not (kind.startswith("phase_") or kind.startswith("solve_")
                                      or kind.endswith(("_start", "_end"))))

    def _text_due(self, kind, level, throttle, data=None):
        if self.verbosity < level:
            return False
        if not self._throttled(kind, throttle):
            return True
        phase = (data or {}).get("phase", self._context.get("phase"))
        key = (self.solver, self.method, kind, phase)
        previous = self._shared.last_output.get(key)
        return previous is None or self._shared.clock()-previous >= self.progress_interval

    def wants(self, kind, level=1, throttle=False):
        """Check demand without consuming an output slot or computing a norm."""
        self.check_precision()
        self._level(level)
        if self._shared.callback is not None and not self._shared.callback_failed:
            return True
        return self._text_due(kind, level, throttle)

    def _format_value(self, value):
        if value is None:
            return "-"
        if isinstance(value, str):
            return value.replace("\n", " ").replace("\r", " ")
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else "no"
        if isinstance(value, tuple):
            return "(" + ", ".join(self._format_value(item) for item in value) + ")"
        if hasattr(value, "_mpf_"):
            owner = self.arithmetic
            context = (owner.context if owner is not None and owner.use_mpmath
                       else value.context)
            return context.nstr(value, n=7)
        if isinstance(value, np.floating):
            return np.format_float_scientific(value, precision=6, unique=False, trim="-")
        if isinstance(value, (float, Decimal)):
            return format(value, ".7g")
        return str(value)

    def _write(self, event):
        label = "/".join(filter(None, (event.solver, event.method)))
        fields = " ".join(name + "=" + self._format_value(value)
                          for name, value in event.data.items())
        prefix = "[{}] ".format(label) if label else ""
        text = "{}{}{} elapsed={:.3f}s".format(
            prefix, event.kind, " " + fields if fields else "", event.elapsed)
        print(text, file=sys.stdout if self._shared.stream is None else self._shared.stream,
              flush=True)

    def emit(self, kind, level=1, throttle=False, **data):
        self.check_precision()
        self._level(level)
        if not isinstance(kind, str) or not kind:
            raise ValueError("progress event kind must be a nonempty string")
        if not self.enabled:
            return
        data = _mapping({**self._context, **data}, self.arithmetic)
        due = self._text_due(kind, level, throttle, data)
        observer = self._shared.callback if not self._shared.callback_failed else None
        if observer is None and not due:
            return
        now = self._shared.clock()
        event = ProgressEvent(kind, int(level), self.solver, self.method,
                              now-self._shared.started, data)
        if observer is not None:
            try:
                observer(event)
            except BaseException as error:
                self._shared.callback_failed = True
                if not isinstance(error, Exception):
                    raise
                self.check_precision()
                raise ProgressCallbackError("progress callback failed: " + str(error)) from error
            else:
                self.check_precision()
        if due:
            self._write(event)
            key = (self.solver, self.method, kind, data.get("phase"))
            self._shared.last_output[key] = now
        self.check_precision()

    @contextmanager
    def phase(self, name, level=1, **data):
        """Report a phase before its work and retain exceptions on completion."""
        if not isinstance(name, str) or not name:
            raise ValueError("phase name must be a nonempty string")
        reporter = self.bind(phase=name, **data)
        started = self._shared.clock()
        reporter.emit("phase_start", level=level)
        try:
            yield reporter
        except BaseException as error:
            try:
                reporter.check_precision()
            except RuntimeError:
                # Error reporting cannot reinterpret native metadata in the
                # changed context or replace the original failure's diagnosis.
                raise error
            reporter.emit("phase_end", level=level, success=False,
                          duration=self._shared.clock()-started,
                          error_type=type(error).__name__, reason=str(error))
            raise
        else:
            reporter.emit("phase_end", level=level, success=True,
                          duration=self._shared.clock()-started)
