"""Positive, fixed composite Gauss--Legendre rules for the physical half-line.

Each plan uses ``x = L (1 + y) / (1 - y)`` and splits at every declared
physical interface before placing any nodes.  The rules are independent of a
discretization or nonlinear solve.  NumPy uses float64 Legendre rules; mpmath
constructs its rules entirely in the owner's cloned context, without float64
nodes, weights, or initial guesses.

Plans are immutable snapshots.  Changing a context's precision invalidates its
old plans; their factory rebuilds from the original mapping and interface inputs
on the next request.  Refinement differences are integration diagnostics, not
rigorous error bounds or a guarantee that an unsplit discontinuity was detected.
"""

from collections import OrderedDict
from dataclasses import dataclass, field
import math
from numbers import Integral
from typing import Any, Tuple

import numpy as np

from rational_mapping import RationalHalfLineMap, arithmetic_signature


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError("{} must be a positive integer".format(name))
    return int(value)


def _finite_scalar(arithmetic, value, name):
    value = arithmetic.scalar(value)
    if not arithmetic.isfinite(value):
        raise ValueError("{} must be finite".format(name))
    return value


@dataclass(frozen=True)
class RefinementCheck:
    """Difference between two rules, compared with ``atol + rtol*|refined|``."""

    coarse_value: Any
    refined_value: Any
    difference: Any
    target: Any
    converged: bool
    refined_plan: Any = field(repr=False, compare=False)


@dataclass(frozen=True)
class QuadraturePlan:
    """A positive common rule, with physical and reference-coordinate weights.

    ``order`` is the number of nodes *per panel*. ``subdivisions`` uniformly
    splits each interface-delimited interval in reference coordinates.  Nodes
    lie strictly inside panels, so neither infinity nor an interface is sampled.
    All node/weight sequences and panel endpoints are immutable scalar tuples.
    """

    order: int
    subdivisions: int
    mapping: Any = field(repr=False, compare=False)
    panels: Tuple[Tuple[Any, Any], ...]
    nodes: Tuple[Any, ...]
    physical_nodes: Tuple[Any, ...]
    weights: Tuple[Any, ...]
    physical_weights: Tuple[Any, ...]
    _factory: Any = field(repr=False, compare=False)

    @property
    def arithmetic(self):
        return self.mapping.arithmetic

    @property
    def precision_signature(self):
        return self.mapping.precision_signature

    @property
    def raw_interfaces(self):
        """Original physical split inputs, including extra integration splits."""
        self.check_precision()
        return self._factory.raw_interfaces

    def check_precision(self):
        """Reject a plan rounded before its owner's precision was changed."""
        self.mapping.check_precision()

    def _integrate(self, callback, nodes, weights):
        self.check_precision()
        if not callable(callback):
            raise TypeError("integrand must be callable")
        arithmetic = self.arithmetic
        products = []
        for node, weight in zip(nodes, weights):
            value = _finite_scalar(arithmetic, callback(node), "integrand value")
            products.append(_finite_scalar(
                arithmetic, weight * value, "weighted integrand value"
            ))
        if arithmetic.use_mpmath:
            value = arithmetic.context.fsum(products)
        else:
            value = math.fsum(products)
        return _finite_scalar(arithmetic, value, "quadrature sum")

    def integrate(self, callback):
        """Approximate ``integral_0^infinity callback(x) dx`` with scalar calls."""
        return self._integrate(callback, self.physical_nodes, self.physical_weights)

    def integrate_reference(self, callback):
        """Approximate ``integral_-1^1 callback(y) dy`` with scalar calls."""
        return self._integrate(callback, self.nodes, self.weights)

    def refined(self, order_factor=2, subdivisions_factor=1):
        """Return the cached rule with multiplied order/panel subdivisions."""
        self.check_precision()
        order_factor = _positive_integer(order_factor, "order_factor")
        subdivisions_factor = _positive_integer(
            subdivisions_factor, "subdivisions_factor"
        )
        if order_factor == subdivisions_factor == 1:
            raise ValueError("refinement must increase order or subdivisions")
        return self._factory.plan(
            self.order * order_factor,
            subdivisions=self.subdivisions * subdivisions_factor,
        )

    def check_refinement(
        self, callback, *, atol, rtol, order_factor=2, subdivisions_factor=1
    ):
        """Compare physical integrals independently of nonlinear convergence.

        The callback must describe the same fixed integrand for both rules.
        Explicit tolerances use the owner's arithmetic, including decimal strings.
        """
        self.check_precision()
        arithmetic = self.arithmetic
        atol = _finite_scalar(arithmetic, atol, "atol")
        rtol = _finite_scalar(arithmetic, rtol, "rtol")
        if atol < 0 or rtol < 0:
            raise ValueError("refinement tolerances must be nonnegative")
        refined = self.refined(order_factor, subdivisions_factor)
        coarse_value = self.integrate(callback)
        refined_value = refined.integrate(callback)
        difference = _finite_scalar(
            arithmetic, abs(refined_value - coarse_value), "refinement difference"
        )
        target = _finite_scalar(
            arithmetic, atol + rtol * abs(refined_value), "refinement target"
        )
        return RefinementCheck(
            coarse_value, refined_value, difference, target,
            bool(difference <= target), refined,
        )


class QuadratureFactory:
    """Build and locally cache rules for one arithmetic and raw geometry.

    ``interfaces`` contains positive, finite physical coordinates in strictly
    increasing order.  Use decimal strings to retain geometry beyond float64.
    Already-rounded mpmath inputs cannot recover discarded digits when a
    context's precision is raised.  No cache is shared across arithmetic owners.
    """

    def __init__(self, arithmetic, L=1, interfaces=(), max_cached_plans=32):
        self._arithmetic = arithmetic
        self._raw_L = L
        self._raw_interfaces = tuple(interfaces)
        self._max_cached_plans = _positive_integer(
            max_cached_plans, "max_cached_plans"
        )
        self._plans = OrderedDict()
        self._rules = OrderedDict()
        self._signature = None
        self._mapping = None
        self._interface_panels = None
        self._physical_panels = None
        self._refresh_geometry()

    @property
    def arithmetic(self):
        return self._arithmetic

    @property
    def raw_interfaces(self):
        """Immutable original physical interface inputs retained for rebuilding."""
        return self._raw_interfaces

    def _refresh_geometry(self):
        signature = arithmetic_signature(self.arithmetic)
        if signature == self._signature:
            return
        mapping = RationalHalfLineMap(self.arithmetic, self._raw_L)
        arithmetic = self.arithmetic
        zero, one = arithmetic.scalar(0), arithmetic.scalar(1)
        boundaries = [-one]
        physical_boundaries = [zero]
        previous = zero
        for raw in self._raw_interfaces:
            physical = _finite_scalar(arithmetic, raw, "physical interface")
            if physical <= previous:
                raise ValueError("physical interfaces must be positive and strictly increasing")
            mapped = mapping.to_reference(physical)
            if not boundaries[-1] < mapped < one:
                raise ValueError("mapped interfaces must be distinct and strictly inside (-1, 1)")
            boundaries.append(mapped)
            physical_boundaries.append(physical)
            previous = physical
        boundaries.append(one)
        physical_boundaries.append(arithmetic.scalar(arithmetic.inf))
        self._mapping = mapping
        self._interface_panels = tuple(zip(boundaries[:-1], boundaries[1:]))
        self._physical_panels = tuple(zip(
            physical_boundaries[:-1], physical_boundaries[1:]
        ))
        self._signature = signature
        self._plans.clear()
        self._rules.clear()

    def _remember(self, cache, key, value):
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._max_cached_plans:
            cache.popitem(last=False)
        return value

    def _gauss_rule(self, order):
        if order in self._rules:
            self._rules.move_to_end(order)
            return self._rules[order]
        arithmetic = self.arithmetic
        if arithmetic.use_mpmath:
            nodes, weights = arithmetic.context.gauss_quadrature(order, "legendre")
        else:
            nodes, weights = np.polynomial.legendre.leggauss(order)
        result = (
            tuple(arithmetic.scalar(node) for node in nodes),
            tuple(arithmetic.scalar(weight) for weight in weights),
        )
        return self._remember(self._rules, order, result)

    def plan(self, order, subdivisions=1):
        """Return a positive rule, rebuilding geometry after precision changes."""
        order = _positive_integer(order, "order")
        subdivisions = _positive_integer(subdivisions, "subdivisions")
        self._refresh_geometry()
        arithmetic, mapping = self.arithmetic, self._mapping
        panels = []
        for left, right in self._interface_panels:
            endpoints = [left]
            for index in range(1, subdivisions):
                endpoints.append(
                    left + (right - left) * arithmetic.scalar(index) / subdivisions
                )
            endpoints.append(right)
            for start, end in zip(endpoints[:-1], endpoints[1:]):
                if not start < end:
                    raise ValueError("reference panel subdivision collapsed at this precision")
                panels.append((start, end))
        panels = tuple(panels)
        key = (self._signature, order, panels)
        if key in self._plans:
            self._plans.move_to_end(key)
            return self._plans[key]
        base_nodes, base_weights = self._gauss_rule(order)
        nodes, weights, physical_nodes, physical_weights = [], [], [], []
        two = arithmetic.scalar(2)
        for panel_index, (left, right) in enumerate(panels):
            physical_left, physical_right = self._physical_panels[
                panel_index // subdivisions
            ]
            midpoint, half_width = (left + right) / two, (right - left) / two
            for base_node, base_weight in zip(base_nodes, base_weights):
                node = _finite_scalar(arithmetic, midpoint + half_width * base_node,
                                      "reference quadrature node")
                if not left < node < right:
                    raise ValueError("quadrature node collapsed onto a panel boundary")
                weight = _finite_scalar(arithmetic, half_width * base_weight,
                                        "reference quadrature weight")
                physical_node = _finite_scalar(arithmetic, mapping.to_physical(node),
                                               "physical quadrature node")
                if not physical_left < physical_node < physical_right:
                    raise ValueError(
                        "physical quadrature node collapsed onto an interface or boundary"
                    )
                physical_weight = _finite_scalar(
                    arithmetic, weight * mapping.jacobian(node),
                    "physical quadrature weight",
                )
                if weight <= 0 or physical_weight <= 0 or physical_node <= 0:
                    raise ValueError("quadrature requires positive weights and physical nodes")
                if nodes and (node <= nodes[-1] or physical_node <= physical_nodes[-1]):
                    raise ValueError("quadrature nodes must remain distinct and increasing")
                nodes.append(node)
                weights.append(weight)
                physical_nodes.append(physical_node)
                physical_weights.append(physical_weight)
        plan = QuadraturePlan(
            order, subdivisions, mapping, panels, tuple(nodes),
            tuple(physical_nodes), tuple(weights), tuple(physical_weights), self,
        )
        return self._remember(self._plans, key, plan)
