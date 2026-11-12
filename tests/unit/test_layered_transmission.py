"""Independent exact transmission identities for all layered material presets."""

from dataclasses import FrozenInstanceError
import unittest

import mpmath as mp

from overscreening_breakdown.arithmetic import MpmathArithmetic, NumpyArithmetic
from overscreening_breakdown.layered import LayeredMaterial, SUPPORTED_INTERFACE_COUNTS, preset_material
from overscreening_breakdown.manufactured.transmission import (
    layered_transmission_case, transmission_case,
)
from overscreening_breakdown.continuation import AffineReference, ResidualHomotopy
from overscreening_breakdown.galerkin.assembly import GalerkinAssembly
from overscreening_breakdown.galerkin.basis import GalerkinBasis
from overscreening_breakdown.quadrature import QuadratureFactory


class LayeredTransmissionIdentityTests(unittest.TestCase):
    def close(self, arithmetic, actual, expected):
        digits = max(8, arithmetic.dps-8) if arithmetic.use_mpmath else 8
        bound = arithmetic.scalar("1e-{}".format(digits))*max(1, abs(actual), abs(expected))
        self.assertLessEqual(abs(actual-expected), bound)

    def test_reference_transmission_and_wall_identities_at_every_declared_interface(self):
        for arithmetic in (NumpyArithmetic(), MpmathArithmetic(30), MpmathArithmetic(120)):
            for kind in ("constant", "graded"):
                for count in SUPPORTED_INTERFACE_COUNTS:
                    with self.subTest(backend=arithmetic.backend, digits=arithmetic.dps,
                                      kind=kind, interfaces=count):
                        material = preset_material(count, kind=kind)
                        case = layered_transmission_case(material)
                        bound = material.bind(arithmetic)
                        physical = case.make_problem().bind(arithmetic)
                        self.assertEqual(len(physical.epsilon.interfaces), count)
                        self.assertEqual(physical.boundary_conditions,
                                         ((1, None, None, -6, None), (0, None, None, None, None)))
                        self.assertEqual(case.u(arithmetic.scalar(0), arithmetic), 1)
                        self.assertEqual(case.derivative(0, arithmetic, 3), -6)
                        for index, point in enumerate(bound.interfaces):
                            self.assertNotEqual(bound.trace(index, "left"), bound.trace(index, "right"))
                            for order in range(3):
                                self.close(arithmetic,
                                           case.interface_derivative(index, arithmetic, order, "left"),
                                           case.interface_derivative(index, arithmetic, order, "right"))
                            left = case.flux_trace(index, arithmetic, "left")
                            right = case.flux_trace(index, arithmetic, "right")
                            self.close(arithmetic, left, right)
                            self.close(arithmetic, case.flux(point, arithmetic), right)
                        # The references exercise transmission with nonzero flux,
                        # rather than a manufactured zero-flux special case.
                        self.assertGreater(abs(case.flux_trace(0, arithmetic, "left")), 0)
                        for value in (case.u(1, arithmetic), case.w(1, arithmetic),
                                      physical.physical_rhs(1, arithmetic.scalar("0.37"))):
                            self.assertIsInstance(value, type(arithmetic.scalar(0)))

    def test_one_interface_reproduces_the_existing_independent_fixture(self):
        a = MpmathArithmetic(120)
        material = LayeredMaterial((("0", "1"), ("1", "2")))
        case, old = layered_transmission_case(material), transmission_case()
        for text in ("0", "0.125", "1", "1.5", "3", "12"):
            x = a.scalar(text)
            for order in range(5):
                self.close(a, case.derivative(x, a, order), old.derivative(x, a, order))
            self.close(a, case.flux(x, a), old.flux(x, a))
            self.close(a, case.make_problem().bind(a).physical_rhs(x, a.scalar("0.37")),
                       old.make_problem().bind(a).physical_rhs(x, a.scalar("0.37")))
        self.close(a, case.flux_trace(0, a), a.scalar(5)/4)

    def test_sources_and_analytic_derivatives_match_independent_differentiation(self):
        for digits in (15, 30, 50, 100, 120):
            a = MpmathArithmetic(digits)
            for kind in ("constant", "graded"):
                for count in (1, 3, 10):
                    for nonlinear in (False, True):
                        with self.subTest(digits=digits, kind=kind, interfaces=count,
                                          nonlinear=nonlinear):
                            material = preset_material(count, kind=kind)
                            case = layered_transmission_case(material, nonlinear=nonlinear)
                            physical = case.make_problem().bind(a)
                            bound = material.bind(a)
                            points = [(left+right)/2 for left, right in
                                      zip(bound.starts, bound.interfaces)]
                            points.append(bound.starts[-1]+a.scalar("0.75"))
                            # Two separated regions give a bounded test of each
                            # formula; all interface identities are checked above.
                            for x in (points[0], points[-1]):
                                for order in (1, 2, 3, 4):
                                    actual = case.derivative(x, a, order)
                                    expected = a.context.diff(lambda t: case.u(t, a), x, order)
                                    self.close(a, actual, expected)
                                manufactured = a.context.diff(lambda t: case.flux(t, a), x)
                                self.close(a, physical.physical_rhs(x, case.u(x, a)), manufactured)
                                u = a.scalar("0.37")
                                # mpmath.diff temporarily raises working
                                # precision. Bind a reference at that precision
                                # instead of mutating a captured bound snapshot.
                                partial = a.context.diff(
                                    lambda v: case.make_problem().bind(a).physical_rhs(x, v), u)
                                self.close(a, physical.physical_rhs_u(x, u), partial)

    def test_binding_keeps_contexts_parameters_and_global_precision_independent(self):
        original = mp.mp.prec
        low, high = MpmathArithmetic(15), MpmathArithmetic(120)
        beta = "0.125"+"0"*77+"1"
        case = layered_transmission_case(preset_material(10, kind="graded"), beta=beta)
        first = case.make_problem().bind(high)
        x, trial = high.scalar("1.125"), high.scalar("0.37")
        before = first.physical_rhs(x, trial)
        case.make_problem().bind(low)
        self.assertEqual(first.physical_rhs(x, trial), before)
        self.assertEqual(first.arithmetic, high)
        self.assertGreater(first.physical_rhs_u(x, high.scalar(1)), high.scalar("1.375"))
        self.assertEqual(mp.mp.prec, original)
        high.context.prec += 1
        with self.assertRaises(RuntimeError):
            first.physical_rhs(x, trial)
        # Explicitly binding again constructs data at the new owner precision.
        rebound = case.make_problem().bind(high)
        self.assertTrue(high.isfinite(rebound.physical_rhs(x, trial)))

    def test_input_validation_and_immutable_case_metadata(self):
        a = MpmathArithmetic(30)
        material = preset_material(1)
        with self.assertRaises(TypeError):
            layered_transmission_case(material, nonlinear=1)
        with self.assertRaises(TypeError):
            layered_transmission_case(object())
        for delta in (0, -1, "nan", "inf"):
            with self.subTest(delta=delta), self.assertRaises(ValueError):
                layered_transmission_case(material, delta=delta).make_problem().bind(a)
        for beta in (-1, "nan", "inf"):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                layered_transmission_case(material, beta=beta).make_problem().bind(a)
        case = layered_transmission_case(material)
        with self.assertRaises(FrozenInstanceError):
            case.material = None
        with self.assertRaises(ValueError):
            case.interface_derivative(0, a, side="middle")
        with self.assertRaises(ValueError):
            case.interface_derivative(1, a)
        with self.assertRaises(ValueError):
            case.interface_derivative(0, a, order=5)

    def test_ten_interface_reference_and_full_jacobian_preserve_every_selected_precision(self):
        for digits in (15, 30, 50, 100, 120):
            with self.subTest(digits=digits):
                a = MpmathArithmetic(digits)
                material = preset_material(10, kind="graded")
                case = layered_transmission_case(material)
                basis = GalerkinBasis(4, a, L="1")
                plan = QuadratureFactory(a, L="1", interfaces=material.bind(a).interfaces).plan(10)
                assembly = GalerkinAssembly(case.make_problem(), basis, plan)
                state = a.vector(["0.125", "-0.0625", "0.03125", "0.3", "-0.2", "0.1", "0.05"])
                direction = a.vector([1, -2, 3, -1, 2, -3, 1])/16
                physical = assembly.nonlinear_system(state)
                matrix, offset = assembly.reference_operator()
                reference = AffineReference(matrix, offset, physical)
                root = reference.solve()
                tolerance = a.scalar("1e-{}".format(max(7, digits-7)))
                self.assertLess(a.norm(reference.residual(root)), tolerance)
                path = ResidualHomotopy(physical, reference)
                q, h = a.scalar("0.375"), a.scalar("0.125")
                derivative = (-path.residual(state+2*h*direction, q)
                              +8*path.residual(state+h*direction, q)
                              -8*path.residual(state-h*direction, q)
                              +path.residual(state-2*h*direction, q))/(12*h)
                # This five-point rule is exact for the cubic reaction, so it
                # independently checks all native digits without tiny steps.
                self.assertLess(a.norm(derivative-path.jacobian(state, q)*direction), tolerance)

