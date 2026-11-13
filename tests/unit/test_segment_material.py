"""Shared finite material presets retain native geometry and notebook behavior."""

from decimal import Decimal, localcontext
import unittest

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import SUPPORTED_INTERFACE_COUNTS, segment_material
from overscreening_breakdown.notebook_examples.segment_galerkin import (
    segment_material as notebook_segment_material,
)


class SegmentMaterialTests(unittest.TestCase):
    def test_notebook_import_reexports_the_shared_preset(self):
        self.assertIs(notebook_segment_material, segment_material)

    def test_every_supported_count_has_ordered_interior_jumps_in_both_styles(self):
        for count in SUPPORTED_INTERFACE_COUNTS:
            for kind in ("constant", "graded"):
                for radius in ("0.5", "12"):
                    with self.subTest(count=count, kind=kind, R=radius):
                        material = segment_material(count, R=radius, kind=kind)
                        self.assertEqual(material.interface_count, count)
                        self.assertEqual(material.kind, kind)
                        self.assertEqual(material.rows[0], ("0", "1"))
                        starts = [Decimal(row[0]) for row in material.rows]
                        self.assertTrue(all(left < right for left, right in zip(starts, starts[1:])))
                        self.assertTrue(all(0 < point < Decimal(radius) for point in starts[1:]))
                        with localcontext() as context:
                            context.prec = 200
                            spacing = Decimal(radius)/Decimal(count+1)
                            for index, start in enumerate(starts[1:], 1):
                                self.assertLess(abs(start-index*spacing), Decimal("1e-176"))
                        levels = ("1", "2", "1.25", "1.75")
                        self.assertEqual(tuple(row[1] for row in material.rows),
                                         tuple(levels[index % 4] for index in range(count+1)))
                        for a in (NumpyArithmetic(), MpmathArithmetic(120)):
                            bound = material.bind(a)
                            self.assertEqual(len(bound.interfaces), count)
                            self.assertTrue(all(a.scalar(0) < point < a.scalar(radius)
                                                for point in bound.interfaces))

    def test_decimal_domain_correction_survives_high_precision_binding(self):
        radius = "1."+"0"*79+"1"
        a = MpmathArithmetic(120)
        material = segment_material(1, R=radius)
        point = material.bind(a).interfaces[0]
        self.assertGreater(point-a.scalar("0.5"), 0)
        self.assertLess(abs(point-a.scalar(radius)/2), a.scalar("1e-118"))
        self.assertEqual(material.to_dict(), notebook_segment_material(1, R=radius).to_dict())

    def test_invalid_radius_count_and_style_fail_before_binding(self):
        for radius in (0, -1, "nan", "inf", "-inf", True, None, "invalid"):
            with self.subTest(R=radius), self.assertRaisesRegex(ValueError, "R must be"):
                segment_material(3, R=radius)
        for count in (0, 4, 6, 11, True, "3", 3.0):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "interfaces"):
                segment_material(count, R=4)
        with self.assertRaisesRegex(ValueError, "material kind"):
            segment_material(3, R=4, kind="invalid")
