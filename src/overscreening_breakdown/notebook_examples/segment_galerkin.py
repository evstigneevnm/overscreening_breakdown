"""Material setup for the guided finite Galerkin notebook.

The physical problem and all solver settings remain visible in the notebook.
These helpers only place and describe material layers; they do not solve or
infer a physical source, boundary condition, or numerical resolution.
"""

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import (
    LayeredMaterial, _segment_radius as _radius, segment_material as segment_material,
)
from overscreening_breakdown.notebook_examples.discontinuous_galerkin import (
    ACCURACY_PRESETS, _coefficient, _number,
)


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
