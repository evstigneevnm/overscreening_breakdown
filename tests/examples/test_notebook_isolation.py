"""Research output stays intact; unknown or heavy cells never enter core CI."""

import copy
import unittest

from tests.examples.notebook_tools import execute_allowed_cells, validate_research_notebook


def document(cells):
    return dict(nbformat=4, nbformat_minor=5, metadata={}, cells=cells)


def code(identifier, source, **extra):
    return dict(cell_type="code", id=identifier, source=source, metadata={},
                execution_count=extra.get("execution_count"), outputs=extra.get("outputs", []))


class NotebookIsolationTests(unittest.TestCase):
    def test_unknown_unguarded_solver_cells_are_skipped_without_execution(self):
        calls = []
        notebook = document([
            code("unknown-heavy", "external_solver(N=100000)\nraise RuntimeError('must not run')"),
            code("core", "N = 100\nanswer = N + 1"),
            code("optional-heavy", "run_extra = True\nexternal_solver(N=50000)"),
        ])
        before = copy.deepcopy(notebook)
        namespace = dict(external_solver=lambda **kwargs: calls.append(kwargs))
        report = execute_allowed_cells(notebook, ("core",), namespace,
                                        controls_by_cell={"core": {"N": 4}})
        self.assertEqual(namespace["answer"], 5)
        self.assertFalse(calls)
        self.assertEqual(report.executed, ("core",))
        self.assertEqual(report.skipped, ("unknown-heavy", "optional-heavy"))
        self.assertEqual(notebook, before)

    def test_bounded_inputs_are_scoped_to_the_selected_cell_and_not_callback_bodies(self):
        notebook = document([
            code("settings", "model = 'original'\nN = 100\ndef callback():\n    N = 100\n    return N"),
            code("other-case", "model = 'other'\nN = 50"),
        ])
        namespace = {}
        execute_allowed_cells(notebook, ("settings", "other-case"), namespace,
                              controls_by_cell={"settings": {"N": 4}})
        self.assertEqual(namespace["N"], 50)
        self.assertEqual(namespace["model"], "other")
        self.assertEqual(namespace["callback"](), 100)
        with self.assertRaises(ValueError):
            execute_allowed_cells(notebook, ("missing",), namespace)

    def test_saved_execution_counts_streams_images_results_and_errors_remain_valid_data(self):
        outputs = [dict(output_type="stream", name="stdout", text=["stored research\n"]),
                   dict(output_type="display_data", data={"image/png": "stored-image-data"}, metadata={}),
                   dict(output_type="execute_result", data={"text/plain": "42"}, metadata={}, execution_count=7),
                   dict(output_type="error", ename="RuntimeError", evalue="saved failed experiment",
                        traceback=["stored traceback"])]
        notebook = document([code("research", "answer = 42", execution_count=7, outputs=outputs)])
        before = copy.deepcopy(notebook)
        validate_research_notebook(notebook)
        self.assertEqual(notebook, before)
        for changes in (dict(execution_count=True), dict(outputs="invalid")):
            invalid = copy.deepcopy(notebook)
            invalid["cells"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_research_notebook(invalid)
        duplicate = document([code("same", "pass"), code("same", "pass")])
        with self.assertRaises(ValueError):
            validate_research_notebook(duplicate)
