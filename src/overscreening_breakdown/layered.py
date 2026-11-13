"""Layer-table material descriptions for discontinuous half-line coefficients.

A row is ``(starts_at_x, epsilon_value)``. The first row starts at zero;
the last material extends to infinity. Decimal strings are retained until
binding, so the same table can be used with float64 or owned mpmath arithmetic.
"""

from bisect import bisect_right
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING, localcontext
from numbers import Integral
from typing import Any, Tuple

from overscreening_breakdown.coefficients import Coefficient, _BoundCoefficient
from overscreening_breakdown.arithmetic import arithmetic_signature, require_precision


SUPPORTED_INTERFACE_COUNTS = (1, 2, 3, 5, 7, 10)


def _decimal(value, name):
    if isinstance(value, bool):
        raise ValueError("{} must be a finite decimal number".format(name))
    try:
        decimal = Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError) as error:
        raise ValueError("{} must be a finite decimal number".format(name)) from error
    if not decimal.is_finite():
        raise ValueError("{} must be a finite decimal number".format(name))
    return decimal


@dataclass(frozen=True, init=False)
class LayeredMaterial:
    """A validated, immutable material table with optional grading in layers.

    ``kind='constant'`` uses the row's value throughout its layer.
    ``kind='graded'`` uses, within layer i,

    ``epsilon(x) = value_i * (1 + grading*(1-exp(-(x-start_i)/grading_length)))``.

    The values at interfaces are right traces. Formula-derived bounds and
    the wall trace are supplied automatically; no sampling estimates them.
    """

    rows: Tuple[tuple, ...]
    kind: str
    grading: str
    grading_length: str

    def __init__(self, layers, *, kind="constant", grading="0.1", grading_length="1"):
        if kind not in ("constant", "graded"):
            raise ValueError("material kind must be 'constant' or 'graded'")
        try:
            raw_rows = tuple(tuple(row) for row in layers)
        except TypeError as error:
            raise ValueError("provide material rows as (start x, epsilon) pairs") from error
        if not raw_rows:
            raise ValueError("provide at least one material row, starting at x=0")
        rows, starts, values = [], [], []
        for index, row in enumerate(raw_rows, 1):
            if len(row) != 2:
                raise ValueError("material row {} must contain start x and epsilon".format(index))
            start = _decimal(row[0], "start x in row {}".format(index))
            value = _decimal(row[1], "epsilon in row {}".format(index))
            if value <= 0:
                raise ValueError("epsilon in row {} must be positive".format(index))
            if index == 1 and start != 0:
                raise ValueError("the first material row must start at x=0")
            if index > 1 and start <= starts[-1]:
                raise ValueError("material starts must increase from left to right; check row {}".format(index))
            rows.append((str(start), str(value)))
            starts.append(start)
            values.append(value)
        grade = _decimal(grading, "grading")
        length = _decimal(grading_length, "grading length")
        if grade < 0:
            raise ValueError("grading must be nonnegative")
        if length <= 0:
            raise ValueError("grading length must be positive")
        if kind == "constant" or grade == 0:
            for index, (left, right) in enumerate(zip(values, values[1:]), 1):
                if left == right:
                    raise ValueError("boundary {} has equal neighboring epsilon values; remove it or change a value".format(index))
        object.__setattr__(self, "rows", tuple(rows))
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "grading", str(grade))
        object.__setattr__(self, "grading_length", str(length))

    @property
    def layers(self):
        return self.rows

    @property
    def interface_count(self):
        return len(self.rows)-1

    @property
    def coefficient(self):
        """Return a coefficient accepted directly by FluxProblem and BSKProblem."""
        return _LayeredCoefficient(self)

    def bind(self, arithmetic):
        return BoundLayeredMaterial(self, arithmetic)

    def to_dict(self):
        """Return JSON-safe settings accepted by ``LayeredMaterial(**settings)``."""
        return dict(layers=[list(row) for row in self.rows], kind=self.kind,
                    grading=self.grading, grading_length=self.grading_length)


@dataclass(frozen=True, init=False)
class BoundLayeredMaterial:
    """Native material values and analytic layer formulas in one arithmetic owner."""

    material: LayeredMaterial = field(repr=False)
    arithmetic: Any = field(repr=False)
    starts: tuple
    interfaces: tuple
    values: tuple
    lower_bound: Any
    upper_bound: Any
    wall_value: Any
    far_value: Any
    grading: Any
    grading_length: Any
    _signature: tuple = field(repr=False)

    def __init__(self, material, arithmetic):
        starts = tuple(arithmetic.scalar(row[0]) for row in material.rows)
        values = tuple(arithmetic.scalar(row[1]) for row in material.rows)
        grade = arithmetic.scalar(material.grading if material.kind == "graded" else 0)
        length = arithmetic.scalar(material.grading_length)
        if (any(not arithmetic.isfinite(value) for value in starts+values+(grade, length))
                or any(value <= 0 for value in values) or length <= 0):
            raise ValueError("material values must stay finite and positive at the selected precision")
        if any(left >= right for left, right in zip(starts, starts[1:])):
            raise ValueError("layer boundaries cannot be distinguished at this precision; use higher precision")
        lower, upper = min(values), max(values)*(1+grade)
        far = values[-1]*(1+grade)
        if not arithmetic.isfinite(upper) or not arithmetic.isfinite(far):
            raise ValueError("material bounds must remain finite at the selected precision")
        for name, value in (
            ("material", material), ("arithmetic", arithmetic), ("starts", starts),
            ("interfaces", starts[1:]), ("values", values), ("lower_bound", lower),
            ("upper_bound", upper), ("wall_value", values[0]), ("far_value", far),
            ("grading", grade), ("grading_length", length),
            ("_signature", arithmetic_signature(arithmetic)),
        ):
            object.__setattr__(self, name, value)
        for index in range(len(self.interfaces)):
            if self.trace(index, "left") == self.trace(index, "right"):
                raise ValueError("boundary {} is not a jump at this precision; change neighboring values or use higher precision".format(index+1))

    def check_precision(self):
        require_precision(self.arithmetic, self._signature,
                          rebuild="bind the material again")

    def _coordinate(self, x):
        self.check_precision()
        x = self.arithmetic.scalar(x)
        if not self.arithmetic.isfinite(x) or x < 0:
            raise ValueError("material coordinate x must be finite and nonnegative")
        return x

    def region_index(self, x):
        return bisect_right(self.interfaces, self._coordinate(x))

    def _region_coordinate(self, index, x):
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.starts):
            raise ValueError("layer index must identify an existing material row")
        x = self._coordinate(x)
        if x < self.starts[index] or (index < len(self.interfaces) and x > self.interfaces[index]):
            raise ValueError("coordinate lies outside the selected layer")
        return x

    def epsilon_in_region(self, index, x):
        x = self._region_coordinate(index, x)
        if self.grading == 0:
            return self.values[index]
        distance = (x-self.starts[index])/self.grading_length
        change = -self.arithmetic.context.expm1(-distance)
        return self.values[index]*(1+self.grading*change)

    def epsilon_dx_in_region(self, index, x):
        x = self._region_coordinate(index, x)
        return (self.values[index]*self.grading/self.grading_length
                *self.arithmetic.exp(-(x-self.starts[index])/self.grading_length))

    def epsilon(self, x):
        return self.epsilon_in_region(self.region_index(x), x)

    __call__ = epsilon

    def epsilon_dx(self, x):
        """Return the ordinary derivative inside a layer, using right traces."""
        return self.epsilon_dx_in_region(self.region_index(x), x)

    def trace(self, index, side="right"):
        self.check_precision()
        if isinstance(index, bool) or not isinstance(index, Integral) or not 0 <= index < len(self.interfaces):
            raise ValueError("interface index must identify an existing positive boundary")
        if side not in ("left", "right"):
            raise ValueError("trace side must be 'left' or 'right'")
        return self.epsilon_in_region(index if side == "left" else index+1,
                                      self.interfaces[index])


class _LayeredCoefficient(Coefficient):
    """Coefficient description that binds table values only once per solver."""

    def __init__(self, material):
        values = tuple(Decimal(row[1]) for row in material.rows)
        grade = Decimal(material.grading) if material.kind == "graded" else Decimal(0)
        precision = max(len(value.as_tuple().digits) for value in values+(grade,))+16
        with localcontext() as context:
            context.prec = 2*precision
            # The descriptor must remain a conservative bound even when a
            # positive grading increment is far below its decimal precision.
            context.rounding = ROUND_CEILING
            upper = max(values)*(1+grade)
        super().__init__(
            lambda x, a: material.bind(a).epsilon(x),
            lower_bound=str(min(values)), upper_bound=str(upper),
            wall_value=material.rows[0][1], interfaces=tuple(row[0] for row in material.rows[1:]),
        )
        object.__setattr__(self, "_material", material)

    def bind(self, arithmetic):
        material = self._material.bind(arithmetic)
        return _BoundCoefficient(
            arithmetic=arithmetic, lower_bound=material.lower_bound,
            upper_bound=material.upper_bound, wall_value=material.wall_value,
            interfaces=material.interfaces, is_identity=False,
            _function=lambda x, a: material.epsilon(x),
        )


def preset_material(interface_count, *, kind="constant", grading="0.1", grading_length="1"):
    """Return a modest-contrast material with exactly the requested jump count."""
    if (isinstance(interface_count, bool) or not isinstance(interface_count, Integral)
            or interface_count not in SUPPORTED_INTERFACE_COUNTS):
        raise ValueError("choose 1, 2, 3, 5, 7, or 10 interfaces")
    count = int(interface_count)
    with localcontext() as context:
        context.prec = 32
        positions = ((Decimal(1),) if count == 1 else
                     tuple(Decimal("0.5")+Decimal("4.5")*index/(count-1)
                           for index in range(count)))
    levels = ("1", "2", "1.25", "1.75")
    rows = [("0", levels[0])]
    rows.extend((str(position), levels[(index+1) % len(levels)])
                for index, position in enumerate(positions))
    return LayeredMaterial(rows, kind=kind, grading=grading, grading_length=grading_length)
