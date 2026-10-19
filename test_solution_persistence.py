"""Regression checks for saved coefficients and solver restarts.

Run with ``python -m unittest test_solution_persistence``.
"""

from contextlib import redirect_stdout
import io
from pathlib import Path
import pickle
import tempfile
import unittest

import mpmath as mp
import numpy as np

from overscreening_breakdown import overscreening_breakdown
from spectral1d import solve_nonlinear_problem


class SolutionPersistenceTests(unittest.TestCase):
    def setUp(self):
        original_precision = mp.mp.prec
        self.addCleanup(setattr, mp.mp, "prec", original_precision)
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="overscreening-solutions-", dir="/tmp"
        )
        self.addCleanup(temporary_directory.cleanup)
        self.directory = Path(temporary_directory.name)
        self.output = io.StringIO()
        redirect = redirect_stdout(self.output)
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def solver(self, use_mpmath=False, precision=60, problem=None):
        options = {"domain": [0, np.inf]} if problem is None else {"problem": problem}
        return solve_nonlinear_problem(
            8,
            L=2,
            tolerance=1e-12,
            use_mpmath=use_mpmath,
            prec=precision,
            use_globalization=False,
            **options,
        )

    def legacy_file(self, coefficients):
        path = self.directory / "legacy.pickle"
        with path.open("wb") as handle:
            pickle.dump(coefficients, handle)
        return path

    def test_numpy_roundtrip_preserves_values_and_float64_dtype(self):
        solver = self.solver()
        coefficients = np.array([-0.0, np.pi, -1e100, 1e-200], dtype=np.float64)
        path = self.directory / "numpy.pickle"

        solver.save_solution(path, coefficients)
        restored = solver.load_data(path)

        self.assertEqual(restored.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(restored, coefficients)
        np.testing.assert_array_equal(np.signbit(restored), np.signbit(coefficients))

    def test_legacy_extended_precision_loads_as_float64(self):
        coefficients = np.array(["1.234567890123456789", "-2.5", "0"], dtype=np.longdouble)
        restored = self.solver().load_data(self.legacy_file(coefficients))

        self.assertEqual(restored.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(restored, coefficients.astype(np.float64))

    def test_legacy_extended_precision_loads_as_mpmath(self):
        coefficients = np.array(["1.234567890123456789", "-2.5", "0"], dtype=np.longdouble)
        restored = self.solver(use_mpmath=True).load_data(self.legacy_file(coefficients))

        self.assertIsInstance(restored, mp.matrix)
        self.assertEqual((restored.rows, restored.cols), (3, 1))
        for actual, value in zip(restored, coefficients):
            numerator, denominator = value.as_integer_ratio()
            self.assertEqual(actual, mp.mpf(numerator) / denominator)

    def test_mpmath_roundtrip_preserves_exact_values_at_lower_precision(self):
        for precision in (15, 30, 50, 100, 120):
            with self.subTest(precision=precision):
                source = self.solver(use_mpmath=True, precision=precision)
                coefficients = mp.matrix([
                    mp.sqrt(2),
                    mp.mpf("1.234567890123456789012345678901234567890123456789"),
                    -mp.pi,
                ])
                original_values = list(coefficients)
                path = self.directory / "mpmath.pickle"
                source.save_solution(path, coefficients)

                destination_precision = min(25, precision)
                destination = self.solver(use_mpmath=True, precision=destination_precision)
                restored = destination.load_data(path)

                self.assertEqual(mp.mp.dps, destination_precision)
                self.assertIsInstance(restored, mp.matrix)
                self.assertEqual((restored.rows, restored.cols), (3, 1))
                self.assertEqual(list(restored), original_values)

    def test_saving_mpmath_values_after_lowering_precision_is_exact(self):
        for precision in (15, 30, 50, 100, 120):
            with self.subTest(precision=precision):
                solver = self.solver(use_mpmath=True, precision=precision)
                coefficients = mp.matrix([mp.sqrt(2), mp.pi, mp.mpf("1e-40")])
                original_values = list(coefficients)
                lowered_precision = min(15, precision - 5)
                mp.mp.dps = lowered_precision
                path = self.directory / "lowered-precision.pickle"
                solver.save_solution(path, coefficients)

                restored = solver.load_data(path)

                self.assertEqual(mp.mp.dps, lowered_precision)
                self.assertEqual(list(restored), original_values)

    def test_numpy_file_can_be_loaded_by_mpmath(self):
        coefficients = np.array([np.pi, -0.125, 1e-20], dtype=np.float64)
        path = self.directory / "numpy-to-mpmath.pickle"
        self.solver().save_solution(path, coefficients)

        restored = self.solver(use_mpmath=True).load_data(path)

        self.assertIsInstance(restored, mp.matrix)
        self.assertEqual((restored.rows, restored.cols), (3, 1))
        self.assertEqual(list(restored), [mp.mpf(float(value)) for value in coefficients])

    def test_mpmath_file_can_be_loaded_by_numpy(self):
        source = self.solver(use_mpmath=True)
        coefficients = mp.matrix([mp.pi, -mp.sqrt(2), mp.mpf("1e-100")])
        expected = np.array([float(value) for value in coefficients], dtype=np.float64)
        path = self.directory / "mpmath-to-numpy.pickle"
        source.save_solution(path, coefficients)

        restored = self.solver().load_data(path)

        self.assertEqual(restored.dtype, np.dtype(np.float64))
        np.testing.assert_array_equal(restored, expected)

    def test_saved_solution_restarts_both_solver_backends(self):
        for use_mpmath in (False, True):
            with self.subTest(use_mpmath=use_mpmath):
                problem = overscreening_breakdown()
                problem.set_parameters(sigma=10.0, mu=0.0, gamma=0.5, delta=1.0, u0=1.0)
                solver = self.solver(use_mpmath=use_mpmath, problem=problem)
                coefficients = solver.solve_problem()
                self.assertTrue(solver.is_converged())
                path = self.directory / ("mp-restart.pickle" if use_mpmath else "np-restart.pickle")
                solver.save_solution(path, coefficients)

                solver.reset_base_solution()
                solver.set_base_solution(path)
                restarted = solver.solve_problem()

                self.assertTrue(solver.is_converged())
                self.assertLess(float(solver.residual_L2_norm(restarted)), 1e-10)
                np.testing.assert_allclose(
                    np.array(list(restarted), dtype=np.float64),
                    np.array(list(coefficients), dtype=np.float64),
                    rtol=1e-10,
                    atol=1e-12,
                )

    def test_unsupported_format_version_is_rejected(self):
        path = self.directory / "unsupported.pickle"
        with path.open("wb") as handle:
            pickle.dump({"format_version": 999, "backend": "numpy", "coefficients": [1.0]}, handle)

        with self.assertRaises(ValueError):
            self.solver().load_data(path)


if __name__ == "__main__":
    unittest.main()
