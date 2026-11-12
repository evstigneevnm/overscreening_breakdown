"""Material setup for the guided finite Galerkin notebook.

The physical problem and all solver settings remain visible in the notebook.
These helpers only place and describe material layers; they do not solve or
infer a physical source, boundary condition, or numerical resolution.
"""

from decimal import Decimal, InvalidOperation, localcontext
from numbers import Integral

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import LayeredMaterial, SUPPORTED_INTERFACE_COUNTS
from overscreening_breakdown.notebook_examples.discontinuous_galerkin import (
    ACCURACY_PRESETS, _coefficient, _number,
)


def _radius(R):
    if isinstance(R, bool):
        raise ValueError("R must be a finite positive decimal number")
    try:
        radius = Decimal(str(R))
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("R must be a finite positive decimal number") from error
    if not radius.is_finite() or radius <= 0:
        raise ValueError("R must be a finite positive decimal number")
    return radius


def segment_material(interface_count, *, R, kind="constant"):
    """Place the requested jumps at k*R/(interface_count+1), k=1,...,m.

    Decimal strings retain at least 180 significant digits for subsequent
    native mpmath binding. Rebuilding this preset with a different R moves
    the interfaces. A domain-length study with fixed physical layers should
    instead reuse its original material description.
    """
    if (isinstance(interface_count, bool) or not isinstance(interface_count, Integral)
            or interface_count not in SUPPORTED_INTERFACE_COUNTS):
        raise ValueError("choose 1, 2, 3, 5, 7, or 10 interfaces")
    radius = _radius(R)
    values = ("1", "2", "1.25", "1.75")
    rows = [("0", values[0])]
    with localcontext() as context:
        context.prec = max(180, len(radius.as_tuple().digits)+40)
        for index in range(1, int(interface_count)+1):
            position = radius*Decimal(index)/Decimal(int(interface_count)+1)
            rows.append((str(position), values[index % len(values)]))
    return LayeredMaterial(rows, kind=kind)


def segment_material_table(material, *, R, accuracy="standard"):
    """Describe only the finite interval, with the final layer ending at R.

    Display values use seven significant digits. The material's immutable
    decimal rows and to_dict() retain its full precision for reuse/export.
    """
    _radius(R)
    if accuracy not in ACCURACY_PRESETS:
        raise ValueError("accuracy must be standard, strict_integration, or high_precision")
    a = MpmathArithmetic(50) if accuracy == "high_precision" else NumpyArithmetic()
    end = a.scalar(R)
    if not a.isfinite(end) or end <= 0:
        raise ValueError("R must stay finite and positive in the selected arithmetic")
    bound = _coefficient(material).bind(a)
    if any(not 0 < point < end for point in bound.interfaces):
        raise ValueError("all material interfaces must lie strictly inside [0,R]")
    heading = "Physical domain: [0, {}]".format(_number(end, a, 7))
    if not isinstance(material, LayeredMaterial):
        return (heading+"\nCustom coefficient | interfaces: {} | bounds: [{}, {}]".format(
            ", ".join(_number(value, a, 7) for value in bound.interfaces),
            _number(bound.lower_bound, a, 7), _number(bound.upper_bound, a, 7)))
    layers = material.bind(a)
    lines = [heading, "Layer | starts at x | ends at x | epsilon at layer start"]
    for index, (start, value) in enumerate(zip(layers.starts, layers.values)):
        stop = layers.starts[index+1] if index+1 < len(layers.starts) else end
        lines.append("{} | {} | {} | {}".format(
            index+1, _number(start, a, 7), _number(stop, a, 7), _number(value, a, 7)))
    lines.append("Interface | epsilon on left | epsilon on right")
    for index, point in enumerate(layers.interfaces):
        lines.append("{} | {} | {}".format(
            _number(point, a, 7), _number(layers.trace(index, "left"), a, 7),
            _number(layers.trace(index, "right"), a, 7)))
    return "\n".join(lines)
