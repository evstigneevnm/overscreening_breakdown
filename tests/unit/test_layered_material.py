"""Layer-table inputs, one-sided traces, and native material arithmetic."""

from dataclasses import FrozenInstanceError
from decimal import Decimal, localcontext
import json
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.coefficients import Coefficient, as_coefficient
from overscreening_breakdown.layered import (
    LayeredMaterial, SUPPORTED_INTERFACE_COUNTS, preset_material,
)
from overscreening_breakdown.problems.flux import FluxProblem


class LayeredMaterialTests(unittest.TestCase):
    def arithmetics(self):
        yield NumpyArithmetic()
        for digits in (15, 30, 50, 100, 120):
            yield MpmathArithmetic(digits)

    def test_all_requested_presets_have_exactly_the_declared_genuine_jumps(self):
        for kind in ("constant", "graded"):
            for count in SUPPORTED_INTERFACE_COUNTS:
                description = preset_material(count, kind=kind)
                self.assertEqual(description.interface_count, count)
                self.assertEqual(len(description.rows), count+1)
                for arithmetic in self.arithmetics():
                    with self.subTest(kind=kind, count=count, digits=arithmetic.dps):
                        material = description.bind(arithmetic)
                        coefficient = description.coefficient.bind(arithmetic)
                        self.assertEqual(len(material.interfaces), count)
                        self.assertEqual(coefficient.interfaces, material.interfaces)
                        self.assertEqual(material.wall_value, 1)
                        self.assertEqual(coefficient(0), 1)
                        for index, position in enumerate(material.interfaces):
                            self.assertNotEqual(material.trace(index, "left"),
                                                material.trace(index, "right"))
                            self.assertEqual(material.epsilon(position), material.trace(index, "right"))
                            self.assertEqual(coefficient(position), material.trace(index, "right"))
                        probes = [arithmetic.scalar(0)]
                        probes.extend((left+right)/2 for left, right in
                                      zip(material.starts, material.starts[1:]))
                        probes.append(material.starts[-1]+10)
                        for point in probes:
                            value = material.epsilon(point)
                            self.assertLessEqual(material.lower_bound, value)
                            self.assertLessEqual(value, material.upper_bound)
                            self.assertEqual(coefficient(point), value)
                            self.assertIsInstance(value, type(arithmetic.scalar(0)))

    def test_graded_formula_and_derivative_use_local_layer_coordinates(self):
        description = LayeredMaterial((("0", "1"), ("2", "3")), kind="graded",
                                      grading="0.25", grading_length="2")
        for arithmetic in self.arithmetics():
            material = description.bind(arithmetic)
            quarter = arithmetic.scalar("0.25")
            exponential = arithmetic.exp(arithmetic.scalar("-0.5"))
            expected = 3*(1+quarter*(1-exponential))
            tolerance = arithmetic.scalar("1e-{}".format(max(12, arithmetic.dps-4)))
            self.assertLess(abs(material.epsilon(3)-expected), tolerance)
            self.assertLess(abs(material.epsilon_dx(3)-3*quarter/2*exponential), tolerance)
            self.assertEqual(material.region_index(2), 1)
            self.assertEqual(material.epsilon(2), 3)
            self.assertEqual(material.trace(0, "right"), 3)
            self.assertEqual(material.epsilon_dx_in_region(1, 2), 3*quarter/2)
            self.assertGreater(material.trace(0, "left"), 1)
            self.assertEqual(material.lower_bound, 1)
            self.assertEqual(material.upper_bound, arithmetic.scalar("3.75"))
            self.assertEqual(material.far_value, arithmetic.scalar("3.75"))

    def test_constant_materials_ignore_grading_and_have_zero_piecewise_derivative(self):
        description = LayeredMaterial(((0, 1), (1, 2)), grading="1000")
        for arithmetic in self.arithmetics():
            material = description.bind(arithmetic)
            self.assertEqual(material.epsilon("0.5"), 1)
            self.assertEqual(material.epsilon(1), 2)
            self.assertEqual(material.epsilon_dx(1), 0)
            self.assertEqual(material.upper_bound, 2)
            self.assertEqual(material.far_value, 2)

    def test_input_errors_are_reported_before_solving(self):
        invalid = (
            ((), "at least one"),
            (((1, 1),), "first material row"),
            (((0, 1), (1, 2), (1, 3)), "increase"),
            (((0, 1), (-1, 2)), "increase"),
            (((0, 0),), "positive"),
            (((0, -1),), "positive"),
            (((0, "nan"),), "finite"),
            (((0, 1), ("inf", 2)), "finite"),
            (((0, 1), (1, 1)), "equal neighboring"),
            (((0, 1, 2),), "start x and epsilon"),
        )
        for rows, message in invalid:
            with self.subTest(rows=rows), self.assertRaisesRegex(ValueError, message):
                LayeredMaterial(rows)
        for options, message in (
            (dict(kind="linear"), "kind"),
            (dict(grading=-1), "nonnegative"),
            (dict(grading="nan"), "finite"),
            (dict(grading_length=0), "positive"),
            (dict(grading_length="inf"), "finite"),
        ):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, message):
                LayeredMaterial(((0, 1),), **options)
        for count in (0, 4, 6, 8, 9, 11, True, "3", 3.0):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "choose 1, 2, 3"):
                preset_material(count)

    def test_boundaries_and_jumps_must_be_resolvable_at_selected_precision(self):
        start = "1."+"0"*79+"1"
        material = LayeredMaterial(((0, 1), (1, 2), (start, 3)))
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30)):
            with self.assertRaisesRegex(ValueError, "higher precision"):
                material.bind(arithmetic)
        high = material.bind(MpmathArithmetic(120))
        self.assertGreater(high.interfaces[1], high.interfaces[0])
        tiny_jump = LayeredMaterial(((0, 1), (1, start)))
        with self.assertRaisesRegex(ValueError, "not a jump at this precision"):
            tiny_jump.bind(NumpyArithmetic())
        self.assertGreater(tiny_jump.bind(MpmathArithmetic(120)).trace(0), 1)

    def test_context_isolation_preserves_sub_float64_material_changes(self):
        original_bits = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_bits)
        description = LayeredMaterial((("0", "1"), ("1", "2")), kind="graded",
                                      grading="1e-80", grading_length="3")
        with localcontext() as context:
            context.prec = 130
            self.assertGreaterEqual(Decimal(description.coefficient.upper_bound),
                                    2*(1+Decimal("1e-80")))
        low_owner, high_owner = MpmathArithmetic(30), MpmathArithmetic(120)
        low, high = description.bind(low_owner), description.bind(high_owner)
        mp.mp.prec = 37
        self.assertEqual(low.epsilon(4), 2)
        self.assertGreater(high.epsilon(4), 2)
        expected = 2*(1+high_owner.scalar("1e-80")*(1-high_owner.exp(-high_owner.scalar(1))))
        self.assertLess(abs(high.epsilon(4)-expected), high_owner.scalar("1e-115"))
        self.assertEqual(high_owner.dps, 120)
        self.assertEqual(low_owner.dps, 30)
        self.assertEqual(mp.mp.prec, 37)
        high_owner.context.prec += 1
        for callback in (lambda: high.epsilon(1), lambda: high.epsilon_dx(1),
                         lambda: high.trace(0), lambda: high.region_index(1)):
            with self.assertRaisesRegex(RuntimeError, "precision changed"):
                callback()

    def test_tables_are_immutable_and_round_trip_through_json_settings(self):
        rows = [["0", "1"], ["1", "2"]]
        description = LayeredMaterial(rows, kind="graded")
        rows[1][1] = "100"
        self.assertEqual(description.rows[1][1], "2")
        self.assertIs(description.rows, description.layers)
        restored = LayeredMaterial(**json.loads(json.dumps(description.to_dict())))
        self.assertEqual(description, restored)
        for target, name, value in (
            (description, "kind", "constant"),
            (description.bind(NumpyArithmetic()), "wall_value", 10),
        ):
            with self.assertRaises(FrozenInstanceError):
                setattr(target, name, value)

    def test_coefficient_integrates_with_physical_problem_without_derivative_callback(self):
        material = preset_material(3, kind="graded")
        coefficient = material.coefficient
        self.assertIsInstance(coefficient, Coefficient)
        self.assertIs(as_coefficient(coefficient), coefficient)
        problem = FluxProblem(delta=1, epsilon=coefficient,
                              f=lambda x, u, a: u, f_u=lambda x, u, a: a.scalar(1))
        for arithmetic in self.arithmetics():
            physical = problem.bind(arithmetic)
            bound = material.bind(arithmetic)
            self.assertEqual(physical.epsilon.interfaces, bound.interfaces)
            self.assertEqual(physical.epsilon.wall_value, bound.wall_value)
            self.assertEqual(physical.epsilon(arithmetic.scalar("1.5")),
                             bound.epsilon(arithmetic.scalar("1.5")))

    def test_coordinate_region_and_trace_errors_are_explicit(self):
        material = preset_material(2).bind(NumpyArithmetic())
        for coordinate in (-1, "nan", "inf"):
            with self.assertRaisesRegex(ValueError, "coordinate"):
                material.epsilon(coordinate)
        for index in (-1, 3, True, 0.5):
            with self.assertRaisesRegex(ValueError, "layer index"):
                material.epsilon_in_region(index, 1)
        with self.assertRaisesRegex(ValueError, "outside"):
            material.epsilon_in_region(0, 2)
        for index in (-1, 2, True):
            with self.assertRaisesRegex(ValueError, "interface index"):
                material.trace(index)
        with self.assertRaisesRegex(ValueError, "trace side"):
            material.trace(0, "middle")

    def test_preset_positions_do_not_depend_on_the_global_decimal_context(self):
        reference = preset_material(10).rows
        with localcontext() as context:
            context.prec = 2
            self.assertEqual(preset_material(10).rows, reference)

