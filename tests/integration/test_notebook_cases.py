"""Named cases keep their actual solution, figures and exports together."""

import csv
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import tempfile
import unittest

from matplotlib import pyplot as plt

from overscreening_breakdown import BSKProblem, ConstantCoefficient, plot_coefficients, plot_fields
from overscreening_breakdown.notebook_examples.cases import CaseArtifacts, export_cases
from overscreening_breakdown.notebook_examples.discontinuous_galerkin import (
    ResolutionComparison, compare_resolution, export_experiment, run_defined_problem,
)
from overscreening_breakdown.notebook_examples.discontinuous_plotting import (
    plot_convergence, plot_experiment, plot_material,
)
from overscreening_breakdown.notebook_examples.homotopy_plotting import plot_homotopy_comparison


def wall_case(wall, *, N=4, use_mpmath=False):
    problem = BSKProblem(
        delta="1", mu="0.05", sigma="1.2", gamma="0.4", epsilon=1, domain=(0, 2),
        boundary_conditions=((wall, None, None, 0, None), (0, 0, None, None, None)),
    )
    return run_defined_problem(
        problem, N=N, newton_atol="1e-20" if use_mpmath else "1e-10", material=ConstantCoefficient(1),
        model="BSK", numerical=dict(use_mpmath=use_mpmath, precision=30),
    )


class NotebookCaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.low, cls.high = wall_case("0.05"), wall_case("0.15")
        assert cls.low.solution.converged and cls.high.solution.converged

    def setUp(self):
        self.addCleanup(plt.close, "all")

    def test_immutable_bundle_copies_its_figure_mapping(self):
        figure = plot_experiment(self.low, samples=5)
        source = {"fields": figure}
        case = CaseArtifacts("low", self.low, figures=source)
        source.clear()
        self.assertIs(case.figures["fields"], figure)
        with self.assertRaises(TypeError):
            case.figures["other"] = figure
        with self.assertRaises(FrozenInstanceError):
            case.name = "changed"

    def test_matching_wall_cases_export_their_own_json_csv_and_pdf(self):
        cases = {name: CaseArtifacts(name, experiment, figures={"fields": plot_experiment(experiment, samples=5)})
                 for name, experiment in (("low", self.low), ("high", self.high))}
        with tempfile.TemporaryDirectory(prefix="overscreening-cases-", dir="/tmp") as temporary:
            paths = export_cases(cases, Path(temporary)/"cases")
            for name, expected_wall in (("low", .05), ("high", .15)):
                exported = paths[name]
                settings = json.loads(exported["settings"].read_text())
                self.assertEqual(settings["case_name"], name)
                self.assertEqual(settings["physical"]["wall_potential"], expected_wall)
                self.assertEqual(settings["figure_provenance"]["fields"]["ownership"], "single_solution")
                provenance = settings["figure_provenance"]["fields"]["solutions"][0]
                self.assertEqual(provenance["boundary_conditions"][0][0], expected_wall)
                self.assertEqual(provenance["arithmetic"]["precision_bits"], 53)
                snapshot = json.loads(exported["solution"].read_text())
                self.assertEqual(snapshot["solver_type"], "galerkin")
                with exported["profiles"].open(newline="") as stream:
                    rows = list(csv.DictReader(stream))
                self.assertAlmostEqual(float(rows[0]["u"]), expected_wall)
                self.assertTrue(exported["fields"].read_bytes().startswith(b"%PDF-"))
            with self.assertRaises(TypeError):
                paths["another"] = paths["low"]
            with self.assertRaises(TypeError):
                paths["low"]["settings"] = Path("changed")

    def test_wrong_owned_figure_preflight_keeps_new_and_existing_destinations_untouched(self):
        wrong = plot_experiment(self.high, samples=5)
        case = CaseArtifacts("low", self.low, figures={"fields": wrong})
        with tempfile.TemporaryDirectory(prefix="overscreening-wrong-case-", dir="/tmp") as temporary:
            root = Path(temporary)/"not-created"
            with self.assertRaisesRegex(ValueError, "different solution"):
                export_cases({"low": case}, root)
            self.assertFalse(root.exists())
            existing = Path(temporary)/"existing"
            existing.mkdir()
            sentinel = existing/"settings.json"
            sentinel.write_text("existing experiment\n")
            with self.assertRaisesRegex(ValueError, "different solution"):
                export_experiment(self.low, existing, figures={"fields": wrong})
            self.assertEqual(sentinel.read_text(), "existing experiment\n")
            self.assertEqual(tuple(existing.iterdir()), (sentinel,))

    def test_registry_wide_preflight_rejects_later_mismatch_before_exporting_first_case(self):
        good = CaseArtifacts("low", self.low)
        wrong = CaseArtifacts("high", self.high, figures={"fields": plot_convergence(self.low)})
        with tempfile.TemporaryDirectory(prefix="overscreening-registry-preflight-", dir="/tmp") as temporary:
            directory = Path(temporary)/"no-partial-registry"
            with self.assertRaisesRegex(ValueError, "different solution"):
                export_cases({"low": good, "high": wrong}, directory)
            self.assertFalse(directory.exists())

    def test_comparison_provenance_allows_members_and_rejects_unrelated_snapshots(self):
        figure = plot_homotopy_comparison({"low": self.low, "high": self.high}, samples=5)
        self.assertEqual(figure._overscreening_solutions, (self.low.solution, self.high.solution))
        with tempfile.TemporaryDirectory(prefix="overscreening-comparison-case-", dir="/tmp") as temporary:
            paths = export_cases({"low": CaseArtifacts("low", self.low, figures={"comparison": figure})}, temporary)
            data = json.loads(paths["low"]["settings"].read_text())
            provenance = data["figure_provenance"]["comparison"]
            self.assertEqual(provenance["ownership"], "comparison")
            self.assertEqual([owner["boundary_conditions"][0][0] for owner in provenance["solutions"]], [.05, .15])
            unrelated = wall_case("0.25")
            forbidden = Path(temporary)/"unrelated"
            with self.assertRaisesRegex(ValueError, "different solution"):
                export_cases({"third": CaseArtifacts("third", unrelated, figures={"comparison": figure})}, forbidden)
            self.assertFalse(forbidden.exists())

    def test_generic_package_plots_and_overlays_keep_all_actual_solution_owners(self):
        unrelated = wall_case("0.25")
        with tempfile.TemporaryDirectory(prefix="overscreening-generic-figure-case-", dir="/tmp") as temporary:
            for factory in (plot_fields, plot_coefficients):
                with self.subTest(plot=factory.__name__):
                    wrong = factory(self.high.solution)
                    directory = Path(temporary)/(factory.__name__+"-wrong")
                    with self.assertRaisesRegex(ValueError, "different solution"):
                        export_cases({"low": CaseArtifacts("low", self.low, figures={"generic": wrong})}, directory)
                    self.assertFalse(directory.exists())
                    overlay = factory(self.low.solution)
                    self.assertIs(factory(self.high.solution, ax=overlay.axes), overlay)
                    self.assertEqual(overlay._overscreening_solutions, (self.low.solution, self.high.solution))
                    exported = export_cases({"low": CaseArtifacts("low", self.low, figures={"overlay": overlay})},
                                            Path(temporary)/(factory.__name__+"-matching"))
                    data = json.loads(exported["low"]["settings"].read_text())
                    provenance = data["figure_provenance"]["overlay"]
                    self.assertEqual(provenance["ownership"], "comparison")
                    self.assertEqual(len(provenance["solutions"]), 2)
                    directory = Path(temporary)/(factory.__name__+"-unrelated")
                    with self.assertRaisesRegex(ValueError, "different solution"):
                        export_cases({"third": CaseArtifacts("third", unrelated,
                                                              figures={"overlay": overlay})}, directory)
                    self.assertFalse(directory.exists())

    def test_metadata_replacement_and_owned_resolution_use_solution_identity(self):
        comparison = compare_resolution(self.low, fine_N=6, newton_atol="1e-10")
        figure = plot_experiment(self.low, samples=5)
        replaced = replace(self.low, settings={**self.low.settings, "view": "metadata only"})
        with tempfile.TemporaryDirectory(prefix="overscreening-case-resolution-", dir="/tmp") as temporary:
            paths = export_cases({"low": CaseArtifacts("low", replaced, figures={"fields": figure},
                                                       resolution=comparison)}, temporary)
            data = json.loads(paths["low"]["settings"].read_text())
            self.assertEqual(data["resolution"]["coarse_N"], 4)
            self.assertEqual(data["resolution"]["fine_N"], 6)
            foreign = ResolutionComparison(self.high, comparison.fine, comparison.relative_changes)
            forbidden = Path(temporary)/"foreign-resolution"
            with self.assertRaisesRegex(ValueError, "belong to this solution"):
                export_cases({"low": CaseArtifacts("low", self.low, resolution=foreign)}, forbidden)
            self.assertFalse(forbidden.exists())

    def test_untagged_material_and_user_figures_remain_explicit_caller_responsibility(self):
        material = plot_material(ConstantCoefficient(1), x_max=2, samples=5)
        custom, axis = plt.subplots()
        axis.plot([0, 1], [0, 1])
        self.assertFalse(hasattr(material, "_overscreening_solutions"))
        with tempfile.TemporaryDirectory(prefix="overscreening-untagged-case-", dir="/tmp") as temporary:
            paths = export_cases({"low": CaseArtifacts("low", self.low, figures={"material": material,
                                                                                     "custom": custom})}, temporary)
            provenance = json.loads(paths["low"]["settings"].read_text())["figure_provenance"]
            self.assertTrue(all(value["ownership"] == "untagged" for value in provenance.values()))
            self.assertTrue(all("caller" in value["note"] for value in provenance.values()))

    def test_invalid_registry_names_and_figure_names_fail_before_paths_exist(self):
        for name in ("../escape", "", "/absolute", "with spaces", "bad\\path"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                CaseArtifacts(name, self.low)
        with tempfile.TemporaryDirectory(prefix="overscreening-invalid-case-", dir="/tmp") as temporary:
            for registry in ({"other": CaseArtifacts("low", self.low)}, {"low": self.low}):
                directory = Path(temporary)/"invalid"
                with self.assertRaises(ValueError):
                    export_cases(registry, directory)
                self.assertFalse(directory.exists())
            figure = plot_convergence(self.low)
            for figures in ({"../escape": figure}, {"settings": figure}, {1: figure, "1": figure}):
                directory = Path(temporary)/"bad-figure"
                with self.assertRaises(ValueError):
                    export_cases({"low": CaseArtifacts("low", self.low, figures=figures)}, directory)
                self.assertFalse(directory.exists())

    def test_stale_snapshot_registry_fails_before_any_directory_is_created(self):
        stale = wall_case("0.2", use_mpmath=True)
        stale.solution.arithmetic.context.prec += 1
        with tempfile.TemporaryDirectory(prefix="overscreening-stale-case-", dir="/tmp") as temporary:
            directory = Path(temporary)/"stale"
            with self.assertRaises(RuntimeError):
                export_cases({"low": CaseArtifacts("low", self.low),
                              "stale": CaseArtifacts("stale", stale)}, directory)
            self.assertFalse(directory.exists())
