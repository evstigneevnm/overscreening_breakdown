"""Read-only research notebook validation and explicitly selected CI cells."""

import ast
from dataclasses import dataclass
import re


def source_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(part, str) for part in value):
        return "".join(value)
    raise ValueError("notebook text must be a string or a list of strings")


def _execution_count(value):
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError("execution_count must be a nonnegative integer or null")


def validate_research_notebook(notebook):
    """Check notebook structure while retaining saved research output as data.

    This intentionally needs no optional nbformat dependency. Source syntax
    and repository-specific public imports are checked separately by callers.
    """
    if (not isinstance(notebook, dict) or type(notebook.get("nbformat")) is not int or notebook["nbformat"] != 4
            or type(notebook.get("nbformat_minor")) is not int or notebook["nbformat_minor"] < 0
            or not isinstance(notebook.get("metadata"), dict)
            or not isinstance(notebook.get("cells"), list)):
        raise ValueError("invalid notebook document structure")
    identifiers = set()
    for cell in notebook["cells"]:
        if (not isinstance(cell, dict) or cell.get("cell_type") not in ("code", "markdown", "raw")
                or not isinstance(cell.get("metadata"), dict)):
            raise ValueError("invalid notebook cell structure")
        source_text(cell.get("source"))
        identifier = cell.get("id")
        if identifier is not None:
            if (not isinstance(identifier, str) or re.fullmatch(r"[A-Za-z0-9_-]{1,64}", identifier) is None
                    or identifier in identifiers):
                raise ValueError("notebook cell IDs must be valid and unique")
            identifiers.add(identifier)
        # Older research tools can leave cells without IDs even after updating
        # the document's minor version. Preserve those files; selected CI cells
        # still require their explicit, unique IDs below.
        if cell["cell_type"] != "code":
            if "attachments" in cell and not isinstance(cell["attachments"], dict):
                raise ValueError("notebook attachments must be an object")
            continue
        if "execution_count" not in cell or not isinstance(cell.get("outputs"), list):
            raise ValueError("code cells require execution_count and an output list")
        _execution_count(cell["execution_count"])
        for output in cell["outputs"]:
            if not isinstance(output, dict):
                raise ValueError("notebook output must be an object")
            kind = output.get("output_type")
            if kind == "stream":
                if output.get("name") not in ("stdout", "stderr"):
                    raise ValueError("invalid notebook stream name")
                source_text(output.get("text"))
            elif kind in ("display_data", "execute_result"):
                if not isinstance(output.get("data"), dict) or not isinstance(output.get("metadata"), dict):
                    raise ValueError("display output requires data and metadata objects")
                if kind == "execute_result":
                    if "execution_count" not in output:
                        raise ValueError("execute_result requires execution_count")
                    _execution_count(output["execution_count"])
            elif kind == "error":
                if (not isinstance(output.get("ename"), str) or not isinstance(output.get("evalue"), str)
                        or not isinstance(output.get("traceback"), list)
                        or not all(isinstance(line, str) for line in output["traceback"])):
                    raise ValueError("invalid notebook error output")
            else:
                raise ValueError("unsupported notebook output type")


def code_tree(source):
    """Parse ordinary Python while leaving inline display magic to its own tests."""
    return ast.parse("\n".join(line for line in source.splitlines()
                               if not line.lstrip().startswith("%")))


def _literal(value):
    expression = ast.parse(repr(value), mode="eval")
    ast.literal_eval(expression)
    return expression.body


def _bounded_tree(cell, controls):
    tree = code_tree(source_text(cell["source"]))
    # Modify only explicit inputs in this cell's top-level statements. Never
    # rewrite callback bodies, branch-local values, or another case's settings.
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                continue
            name = statement.targets[0].id
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            name = statement.target.id
        else:
            continue
        if name in controls:
            statement.value = ast.copy_location(_literal(controls[name]), statement.value or statement)
    return ast.fix_missing_locations(tree)


@dataclass(frozen=True)
class NotebookExecution:
    executed: tuple
    skipped: tuple


def execute_allowed_cells(notebook, allowed_ids, namespace, *, controls_by_cell=None, filename="notebook"):
    """Execute only named code cells, preserving the notebook document itself.

    Unknown and optional cells are reported as skipped, regardless of their
    source contents or saved switches. Missing requested cells fail before any
    execution. Controls belong to explicit cells and named top-level inputs.
    """
    validate_research_notebook(notebook)
    allowed = tuple(allowed_ids)
    if len(allowed) != len(set(allowed)):
        raise ValueError("CI cell allowlist must contain unique IDs")
    by_id = {cell.get("id"): cell for cell in notebook["cells"] if cell["cell_type"] == "code"}
    missing = set(allowed)-by_id.keys()
    if missing:
        raise ValueError("missing approved notebook cells: {}".format(sorted(missing)))
    controls = controls_by_cell or {}
    if set(controls)-set(allowed):
        raise ValueError("bounded controls must name explicitly selected cells")
    executed, skipped = [], []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        identifier = cell.get("id")
        if identifier not in allowed:
            skipped.append(identifier)
            continue
        tree = _bounded_tree(cell, controls.get(identifier, {}))
        exec(compile(tree, filename, "exec"), namespace)
        executed.append(identifier)
    return NotebookExecution(tuple(executed), tuple(skipped))
