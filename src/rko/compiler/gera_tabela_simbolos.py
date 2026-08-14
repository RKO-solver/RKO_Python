"""Source resolution and typed-AST construction for RKO Native Language V1."""

from __future__ import annotations

import ast
import inspect
import math
from pathlib import Path
import re
import textwrap
from typing import Any

from .model import (
    DataFieldIR,
    Diagnostic,
    EnvironmentIR,
    FunctionIR,
    NativeCompileError,
    NativeType,
    ParameterIR,
    SourceSpan,
)


_SCALAR_ANNOTATIONS = {
    "bool": NativeType.bool(),
    "int": NativeType.int(),
    "float": NativeType.float(),
    "None": NativeType.none(),
}


def cpp_identifier(name: str) -> str:
    rendered = re.sub(r"[^A-Za-z0-9_]", "_", name)
    if not rendered or rendered[0].isdigit():
        rendered = f"_{rendered}"
    return rendered


def parse_annotation(node: ast.expr | None, filename: str) -> NativeType:
    if node is None:
        raise NativeCompileError(
            Diagnostic(
                "RKO-NATIVE-TYPE-001",
                "V1 requires an annotation for every parameter and return value",
            )
        )
    if isinstance(node, ast.Name) and node.id in _SCALAR_ANNOTATIONS:
        return _SCALAR_ANNOTATIONS[node.id]
    if isinstance(node, ast.Constant) and node.value is None:
        return NativeType.none()
    if isinstance(node, ast.Subscript):
        root = _annotation_root(node.value)
        if root in {"list", "typing.List", "List"}:
            return NativeType.list(parse_annotation(node.slice, filename))
        if root in {"tuple", "typing.Tuple", "Tuple"}:
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            return NativeType.tuple(
                tuple(parse_annotation(element, filename) for element in elements)
            )
    raise NativeCompileError(
        Diagnostic(
            "RKO-NATIVE-TYPE-002",
            f"unsupported V1 annotation: {ast.unparse(node)}",
            _span(filename, node),
            adaptation="use bool, int, float, list[T], tuple[...] or None",
        )
    )


def _annotation_root(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _annotation_root(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    return None


def runtime_type(
    value: Any, *, path: str, _active_containers: set[int] | None = None
) -> NativeType:
    if _active_containers is None:
        _active_containers = set()
    if isinstance(value, bool):
        return NativeType.bool()
    if isinstance(value, int):
        if value < -(2**63) or value > 2**63 - 1:
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-DATA-001",
                    f"{path} is outside the V1 int64 range",
                )
            )
        return NativeType.int()
    if isinstance(value, float):
        if not math.isfinite(value):
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-DATA-002",
                    f"{path} contains a non-finite float",
                    adaptation="normalize NaN/inf before native compilation",
                )
            )
        return NativeType.float()
    if isinstance(value, list):
        identity = id(value)
        if identity in _active_containers:
            raise NativeCompileError(
                Diagnostic("RKO-NATIVE-DATA-010", f"cyclic container detected at {path}")
            )
        if not value:
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-DATA-003",
                    f"cannot infer the element type of empty runtime field {path}",
                    adaptation="provide a non-empty typed field in V1",
                )
            )
        _active_containers.add(identity)
        try:
            item_types = [
                runtime_type(
                    item,
                    path=f"{path}[{index}]",
                    _active_containers=_active_containers,
                )
                for index, item in enumerate(value)
            ]
        finally:
            _active_containers.remove(identity)
        first = item_types[0]
        if any(item_type != first for item_type in item_types[1:]):
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-DATA-004",
                    f"{path} is not a homogeneous V1 list",
                    adaptation="replace heterogeneous rows with parallel homogeneous arrays or tuples",
                )
            )
        return NativeType.list(first)
    if isinstance(value, tuple):
        identity = id(value)
        if identity in _active_containers:
            raise NativeCompileError(
                Diagnostic("RKO-NATIVE-DATA-010", f"cyclic container detected at {path}")
            )
        _active_containers.add(identity)
        try:
            return NativeType.tuple(
                tuple(
                    runtime_type(
                        item,
                        path=f"{path}[{index}]",
                        _active_containers=_active_containers,
                    )
                    for index, item in enumerate(value)
                )
            )
        finally:
            _active_containers.remove(identity)
    if type(value).__module__.startswith("numpy"):
        if type(value).__name__ == "ndarray":
            normalized = value.tolist()
            return runtime_type(normalized, path=path, _active_containers=_active_containers)
        if hasattr(value, "item"):
            return runtime_type(value.item(), path=path, _active_containers=_active_containers)
    raise NativeCompileError(
        Diagnostic(
            "RKO-NATIVE-DATA-005",
            f"unsupported runtime value at {path}: {type(value).__qualname__}",
            adaptation="normalize it to numeric scalars, homogeneous lists or tuples",
        )
    )


def normalize_runtime_value(value: Any) -> Any:
    if type(value).__module__.startswith("numpy"):
        if type(value).__name__ == "ndarray":
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
    if isinstance(value, list):
        return [normalize_runtime_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_runtime_value(item) for item in value)
    return value


def _contains_list(type_info: NativeType) -> bool:
    if type_info.kind == "list":
        return True
    return type_info.kind == "tuple" and any(_contains_list(item) for item in type_info.items)


def _contains_none(type_info: NativeType) -> bool:
    if type_info.kind == "none":
        return True
    if type_info.kind == "list" and type_info.item is not None:
        return _contains_none(type_info.item)
    return type_info.kind == "tuple" and any(_contains_none(item) for item in type_info.items)


def import_aliases(source_file: Path) -> dict[str, str]:
    tree = ast.parse(source_file.read_text(encoding="utf-8"), filename=str(source_file))
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for imported in node.names:
                aliases[imported.asname or imported.name.split(".")[0]] = imported.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for imported in node.names:
                aliases[imported.asname or imported.name] = f"{node.module}.{imported.name}"
    return aliases


def _span(filename: str, node: ast.AST) -> SourceSpan:
    return SourceSpan(
        filename=filename,
        line=getattr(node, "lineno", 0),
        column=getattr(node, "col_offset", 0),
        end_line=getattr(node, "end_lineno", None),
        end_column=getattr(node, "end_col_offset", None),
    )


class _MethodDependencyVisitor(ast.NodeVisitor):
    def __init__(self, method_names: set[str]) -> None:
        self.method_names = method_names
        self.dependencies: list[str] = []

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in self.method_names
        ):
            if node.func.attr not in self.dependencies:
                self.dependencies.append(node.func.attr)
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)


class _SelfFieldVisitor(ast.NodeVisitor):
    def __init__(self, method_names: set[str], filename: str, call_chain: tuple[str, ...]) -> None:
        self.method_names = method_names
        self.filename = filename
        self.call_chain = call_chain
        self.fields: set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
            and node.func.attr in self.method_names
        ):
            for argument in node.args:
                self.visit(argument)
            for keyword in node.keywords:
                self.visit(keyword.value)
            return
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if isinstance(node.ctx, ast.Store):
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-EFFECT-001",
                        f"mutation of self.{node.attr} is forbidden in V1",
                        _span(self.filename, node),
                        self.call_chain,
                        "move evaluation state to an annotated local list or scalar",
                    )
                )
            if node.attr not in self.method_names:
                self.fields.add(node.attr)
        self.generic_visit(node)


def _root_name(node: ast.expr) -> str | None:
    while isinstance(node, ast.Subscript):
        node = node.value
    if isinstance(node, ast.Name):
        return node.id
    return None


def _is_frozen_self_expression(node: ast.expr) -> bool:
    if isinstance(node, ast.Attribute):
        return isinstance(node.value, ast.Name) and node.value.id == "self"
    if isinstance(node, ast.Subscript):
        return _is_frozen_self_expression(node.value)
    return False


class _FrozenMutationVisitor(ast.NodeVisitor):
    def __init__(self, filename: str, function_name: str) -> None:
        self.filename = filename
        self.call_chain = (function_name,)

    def fail(self, node: ast.AST, field_text: str) -> None:
        raise NativeCompileError(
            Diagnostic(
                "RKO-NATIVE-EFFECT-001",
                f"mutation of frozen environment data {field_text} is forbidden in V1",
                _span(self.filename, node),
                self.call_chain,
                "copy the data to an annotated local list before mutating it",
            )
        )

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            if isinstance(target, ast.Subscript) and _is_frozen_self_expression(target.value):
                self.fail(target, ast.unparse(target.value))
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if isinstance(node.target, ast.Subscript) and _is_frozen_self_expression(node.target.value):
            self.fail(node.target, ast.unparse(node.target.value))
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in {"append", "pop", "clear"}
            and _is_frozen_self_expression(node.func.value)
        ):
            self.fail(node, ast.unparse(node.func.value))
        self.generic_visit(node)


class _DirectMutationVisitor(ast.NodeVisitor):
    def __init__(self, parameter_names: set[str]) -> None:
        self.parameter_names = parameter_names
        self.mutated: set[str] = set()

    def _mark_target(self, target: ast.expr) -> None:
        name = _root_name(target)
        if name in self.parameter_names and isinstance(target, ast.Subscript):
            self.mutated.add(name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._mark_target(target)
        self.generic_visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._mark_target(node.target)
        self.generic_visit(node.value)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Attribute) and node.func.attr in {"append", "pop", "clear"}:
            name = _root_name(node.func.value)
            if name in self.parameter_names:
                self.mutated.add(name)
        self.generic_visit(node)


class _FunctionTyper:
    """Type-check a V1 function and retain a typed lexical environment."""

    _BINOPS = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
    )

    def __init__(
        self,
        *,
        filename: str,
        function_name: str,
        parameters: list[ParameterIR],
        return_type: NativeType,
        fields: dict[str, NativeType],
        signatures: dict[str, tuple[list[NativeType], NativeType]],
        method_names: set[str],
        aliases: dict[str, str],
    ) -> None:
        self.filename = filename
        self.function_name = function_name
        self.call_chain = (function_name,)
        self.return_type = return_type
        self.fields = fields
        self.signatures = signatures
        self.method_names = method_names
        self.aliases = aliases
        self.locals: dict[str, NativeType] = {parameter.name: parameter.type for parameter in parameters}
        self.initialized: set[str] = set(self.locals)
        self.parameter_names = set(self.locals)
        self.loop_targets: set[str] = set()
        self.empty_list_types: dict[int, NativeType] = {}
        self.loop_depth = 0

    def error(self, node: ast.AST, code: str, message: str, adaptation: str | None = None) -> None:
        raise NativeCompileError(
            Diagnostic(code, message, _span(self.filename, node), self.call_chain, adaptation)
        )

    @staticmethod
    def _assignable(expected: NativeType, actual: NativeType) -> bool:
        return expected == actual

    def _check(self, node: ast.AST, expected: NativeType, actual: NativeType) -> None:
        if not self._assignable(expected, actual):
            self.error(
                node,
                "RKO-NATIVE-TYPE-003",
                f"expected {expected.display()}, received {actual.display()}",
            )

    def analyze(
        self, node: ast.FunctionDef
    ) -> tuple[dict[str, NativeType], set[str], dict[int, NativeType]]:
        self.block(node.body)
        return self.locals, self.loop_targets, self.empty_list_types

    def block(self, statements: list[ast.stmt]) -> None:
        for statement in statements:
            self.statement(statement)

    def statement(self, node: ast.stmt) -> None:
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            expected = self._target_type(target)
            value_type = self.expression(node.value, expected=expected)
            if value_type == NativeType.none():
                self.error(node.value, "RKO-NATIVE-TYPE-025", "a None/void call result cannot be assigned in V1")
            if (
                isinstance(target, ast.Name)
                and target.id in self.parameter_names
                and value_type.kind == "list"
            ):
                self.error(
                    target,
                    "RKO-NATIVE-EFFECT-013",
                    f"list parameter {target.id} cannot be rebound in V1",
                    "assign the owned value to a fresh annotated local",
                )
            self._check_list_ownership_assignment(node.value, value_type)
            self._bind_target(target, value_type)
            return
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            declared = parse_annotation(node.annotation, self.filename)
            if declared == NativeType.none():
                self.error(node.annotation, "RKO-NATIVE-TYPE-026", "None is only valid as a V1 helper return type")
            if node.target.id in self.parameter_names and declared.kind == "list":
                self.error(node.target, "RKO-NATIVE-EFFECT-013", f"list parameter {node.target.id} cannot be rebound in V1")
            if node.value is not None:
                actual = self.expression(node.value, expected=declared)
                if actual == NativeType.none():
                    self.error(node.value, "RKO-NATIVE-TYPE-025", "a None/void call result cannot be assigned in V1")
                self._check(node.value, declared, actual)
                self._check_list_ownership_assignment(node.value, actual)
                self._bind_name(node.target, declared)
            else:
                previous = self.locals.get(node.target.id)
                if previous is not None:
                    self._check(node.target, previous, declared)
                else:
                    self.locals[node.target.id] = declared
            return
        if isinstance(node, ast.AugAssign):
            if not isinstance(
                node.op,
                (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod),
            ):
                self.error(node, "RKO-NATIVE-SYNTAX-012", "unsupported V1 augmented-assignment operator")
            target_type = self._required_target_type(node.target)
            value_type = self.expression(node.value)
            if not _is_numeric(target_type) or not _is_numeric(value_type):
                self.error(node, "RKO-NATIVE-TYPE-004", "augmented assignment requires numeric operands")
            if isinstance(node.op, ast.Div) and target_type != NativeType.float():
                self.error(
                    node,
                    "RKO-NATIVE-TYPE-019",
                    "`/=` changes an integer variable to float in Python and is not allowed on int in V1",
                    "use a float local variable or write an explicit float expression",
                )
            result_type = (
                NativeType.float()
                if isinstance(node.op, ast.Div)
                or NativeType.float() in {target_type, value_type}
                else NativeType.int()
            )
            self._check(node, target_type, result_type)
            return
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            self.expression(node.value)
            return
        if isinstance(node, ast.If):
            self._check(node.test, NativeType.bool(), self.expression(node.test))
            initialized_before = set(self.initialized)
            self.block(node.body)
            initialized_body = set(self.initialized)
            self.initialized = set(initialized_before)
            self.block(node.orelse)
            initialized_else = set(self.initialized)
            self.initialized = initialized_body & initialized_else
            return
        if isinstance(node, ast.While):
            self._check(node.test, NativeType.bool(), self.expression(node.test))
            initialized_before = set(self.initialized)
            self.loop_depth += 1
            self.block(node.body)
            self.loop_depth -= 1
            self.initialized = initialized_before
            if node.orelse:
                self.error(node, "RKO-NATIVE-SYNTAX-001", "while/else is outside V1")
            return
        if isinstance(node, ast.For):
            target_type = self._iterated_type(node.iter)
            initialized_before = set(self.initialized)
            self._bind_loop_target(node.target, target_type)
            self.loop_depth += 1
            self.block(node.body)
            self.loop_depth -= 1
            self.initialized = initialized_before
            if node.orelse:
                self.error(node, "RKO-NATIVE-SYNTAX-002", "for/else is outside V1")
            return
        if isinstance(node, ast.Return):
            if node.value is None:
                actual = NativeType.none()
            else:
                actual = self.expression(node.value, expected=self.return_type)
            self._check(node, self.return_type, actual)
            return
        if isinstance(node, (ast.Break, ast.Continue)):
            if self.loop_depth == 0:
                self.error(
                    node,
                    "RKO-NATIVE-SYNTAX-011",
                    f"{type(node).__name__.lower()} is only valid inside a V1 loop",
                )
            return
        if isinstance(node, ast.Pass):
            return
        self.error(
            node,
            "RKO-NATIVE-SYNTAX-003",
            f"unsupported V1 statement: {type(node).__name__}",
        )

    def _target_type(self, target: ast.expr) -> NativeType | None:
        if isinstance(target, ast.Name):
            return self.locals.get(target.id)
        if isinstance(target, ast.Subscript):
            return self._required_target_type(target)
        return None

    def _required_target_type(self, target: ast.expr) -> NativeType:
        if isinstance(target, ast.Name):
            if target.id in self.loop_targets:
                self.error(
                    target,
                    "RKO-NATIVE-EFFECT-016",
                    f"loop target {target.id} is read-only in V1",
                    "assign the value to a fresh local name",
                )
            if target.id not in self.locals or target.id not in self.initialized:
                self.error(target, "RKO-NATIVE-TYPE-005", f"uninitialized variable {target.id}")
            return self.locals[target.id]
        if isinstance(target, ast.Subscript):
            container = self.expression(target.value)
            if container.kind == "list" and container.item is not None:
                self._check(target.slice, NativeType.int(), self.expression(target.slice))
                return container.item
            self.error(target, "RKO-NATIVE-TYPE-006", "subscript assignment requires a list")
        self.error(target, "RKO-NATIVE-SYNTAX-004", "only names and list subscripts may be assignment targets")
        raise AssertionError("unreachable")

    def _bind_target(self, target: ast.expr, value_type: NativeType) -> None:
        if isinstance(target, ast.Name):
            self._bind_name(target, value_type)
            return
        expected = self._required_target_type(target)
        self._check(target, expected, value_type)

    def _bind_name(self, node: ast.Name, value_type: NativeType) -> None:
        if node.id in self.loop_targets:
            self.error(
                node,
                "RKO-NATIVE-EFFECT-016",
                f"loop target {node.id} is read-only and cannot be rebound in V1",
                "assign the value to a fresh local name",
            )
        previous = self.locals.get(node.id)
        if previous is not None:
            self._check(node, previous, value_type)
        else:
            self.locals[node.id] = value_type
        self.initialized.add(node.id)

    def _check_list_ownership_assignment(
        self, node: ast.expr, value_type: NativeType
    ) -> None:
        if value_type.kind != "list":
            return
        if not self._is_owned_list_expression(node):
            self.error(
                node,
                "RKO-NATIVE-EFFECT-003",
                "a local list must be created by an explicit V1 owning expression",
                "use `.copy()` for a flat owned list, or pass the list directly to a read-only helper",
            )

    def _is_owned_list_expression(self, node: ast.expr) -> bool:
        if isinstance(node, (ast.List, ast.ListComp)):
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            return isinstance(node.left, ast.List)
        if isinstance(node, ast.IfExp):
            return self._is_owned_list_expression(node.body) and self._is_owned_list_expression(node.orelse)
        if not isinstance(node, ast.Call):
            return False
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "copy"
        ):
            return True
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "tolist"
            and isinstance(node.func.value, ast.Call)
        ):
            return self._canonical_name(node.func.value.func) in {"numpy.argsort", "np.argsort"}
        return self._canonical_name(node.func) in {"numpy.argsort", "np.argsort"}

    def _bind_loop_target(self, target: ast.expr, item_type: NativeType) -> None:
        if isinstance(target, ast.Name):
            if target.id in self.initialized or target.id in self.loop_targets:
                self.error(
                    target,
                    "RKO-NATIVE-EFFECT-004",
                    f"loop target {target.id} would overwrite an initialized local",
                    "use a fresh loop-variable name",
                )
            self.locals[target.id] = item_type
            self.initialized.add(target.id)
            self.loop_targets.add(target.id)
            return
        if isinstance(target, ast.Tuple) and item_type.kind == "tuple" and len(target.elts) == len(item_type.items):
            for element, element_type in zip(target.elts, item_type.items):
                if not isinstance(element, ast.Name):
                    self.error(target, "RKO-NATIVE-SYNTAX-005", "loop unpack targets must be names")
                if element.id in self.initialized or element.id in self.loop_targets:
                    self.error(
                        element,
                        "RKO-NATIVE-EFFECT-004",
                        f"loop target {element.id} would overwrite an initialized local",
                        "use fresh loop-variable names",
                    )
                self.locals[element.id] = element_type
                self.initialized.add(element.id)
                self.loop_targets.add(element.id)
            return
        self.error(target, "RKO-NATIVE-SYNTAX-006", "unsupported for-loop target")

    def _iterated_type(self, node: ast.expr) -> NativeType:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
            if not 1 <= len(node.args) <= 3 or node.keywords:
                self.error(node, "RKO-NATIVE-CALL-001", "range expects one to three positional arguments")
            for argument in node.args:
                self._check(argument, NativeType.int(), self.expression(argument))
            if len(node.args) == 3:
                step = node.args[2]
                if not isinstance(step, ast.Constant) or isinstance(step.value, bool) or not isinstance(step.value, int):
                    self.error(
                        step,
                        "RKO-NATIVE-CALL-018",
                        "V1 range step must be a constant integer",
                    )
                if step.value == 0:
                    self.error(step, "RKO-NATIVE-CALL-019", "range step cannot be zero")
            return NativeType.int()
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "enumerate":
            if len(node.args) != 1 or node.keywords:
                self.error(node, "RKO-NATIVE-CALL-002", "enumerate expects one positional argument")
            container = self.expression(node.args[0])
            if container.kind != "list" or container.item is None:
                self.error(node, "RKO-NATIVE-TYPE-007", "enumerate requires a list")
            if container.item.kind in {"list", "tuple"}:
                self.error(node, "RKO-NATIVE-EFFECT-005", "iteration over nested mutable values is outside V1")
            return NativeType.tuple((NativeType.int(), container.item))
        iterable = self.expression(node)
        if iterable.kind != "list" or iterable.item is None:
            self.error(node, "RKO-NATIVE-TYPE-008", "V1 for-loops require range, enumerate or a list")
        if iterable.item.kind in {"list", "tuple"}:
            self.error(node, "RKO-NATIVE-EFFECT-005", "iteration over nested mutable values is outside V1")
        return iterable.item

    def expression(self, node: ast.expr, expected: NativeType | None = None) -> NativeType:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return NativeType.bool()
            if isinstance(node.value, int):
                if node.value < -(2**63) or node.value > 2**63 - 1:
                    self.error(node, "RKO-NATIVE-VALUE-004", "integer literal is outside the V1 int64 range")
                return NativeType.int()
            if isinstance(node.value, float):
                if not math.isfinite(node.value):
                    self.error(node, "RKO-NATIVE-VALUE-001", "non-finite float literals are outside V1")
                return NativeType.float()
            if node.value is None:
                return NativeType.none()
            self.error(node, "RKO-NATIVE-VALUE-002", f"unsupported literal {node.value!r}")
        if isinstance(node, ast.Name):
            if node.id not in self.locals or node.id not in self.initialized:
                self.error(node, "RKO-NATIVE-NAME-001", f"uninitialized local name {node.id}")
            return self.locals[node.id]
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr not in self.fields:
                self.error(node, "RKO-NATIVE-DATA-006", f"self.{node.attr} is not a frozen data field")
            return self.fields[node.attr]
        if isinstance(node, ast.List):
            if not node.elts:
                if expected is None or expected.kind != "list":
                    self.error(
                        node,
                        "RKO-NATIVE-TYPE-009",
                        "an empty list needs an explicit local annotation",
                        "write `values: list[T] = []`",
                    )
                self.empty_list_types[id(node)] = expected
                return expected
            item_types = [self.expression(element) for element in node.elts]
            item_type = item_types[0]
            for element, other in zip(node.elts[1:], item_types[1:]):
                self._check(element, item_type, other)
            if item_type.kind in {"list", "tuple"}:
                self.error(
                    node,
                    "RKO-NATIVE-EFFECT-006",
                    "nested list/tuple literals have Python alias semantics outside V1",
                    "build nested evaluation state explicitly with scalar append operations",
                )
            return NativeType.list(item_type)
        if isinstance(node, ast.Tuple):
            result = NativeType.tuple(tuple(self.expression(element) for element in node.elts))
            if _contains_list(result):
                self.error(
                    node,
                    "RKO-NATIVE-EFFECT-015",
                    "tuples containing lists can hide mutable aliases and are outside V1",
                )
            return result
        if isinstance(node, ast.ListComp):
            if len(node.generators) != 1 or node.generators[0].ifs or node.generators[0].is_async:
                self.error(
                    node,
                    "RKO-NATIVE-SYNTAX-008",
                    "V1 list comprehensions require one synchronous generator without filters",
                )
            generator = node.generators[0]
            iterable = self.expression(generator.iter)
            if iterable.kind != "list" or iterable.item is None or not isinstance(generator.target, ast.Name):
                self.error(node, "RKO-NATIVE-TYPE-018", "V1 list comprehension requires a list and a name target")
            if iterable.item.kind in {"list", "tuple"}:
                self.error(node, "RKO-NATIVE-EFFECT-005", "list comprehension over nested mutable values is outside V1")
            if generator.target.id in self.initialized or generator.target.id in self.loop_targets:
                self.error(
                    generator.target,
                    "RKO-NATIVE-EFFECT-007",
                    "a V1 comprehension target cannot shadow an initialized local",
                )
            self.locals[generator.target.id] = iterable.item
            self.initialized.add(generator.target.id)
            self.loop_targets.add(generator.target.id)
            element = self.expression(node.elt)
            self.initialized.remove(generator.target.id)
            if element.kind in {"list", "tuple"}:
                self.error(node, "RKO-NATIVE-EFFECT-006", "nested comprehension results are outside V1")
            return NativeType.list(element)
        if isinstance(node, ast.Subscript):
            container = self.expression(node.value)
            if isinstance(node.slice, ast.Slice):
                self.error(
                    node,
                    "RKO-NATIVE-SYNTAX-009",
                    "list slices are outside V1",
                    "copy the complete list with .copy() or rewrite the slice as an explicit loop",
                )
            index_type = self.expression(node.slice)
            self._check(node.slice, NativeType.int(), index_type)
            if container.kind == "list" and container.item is not None:
                return container.item
            if container.kind == "tuple" and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, int):
                index = node.slice.value
                normalized = index if index >= 0 else len(container.items) + index
                if normalized < 0 or normalized >= len(container.items):
                    self.error(node.slice, "RKO-NATIVE-VALUE-003", "tuple index is outside its fixed bounds")
                return container.items[normalized]
            self.error(node, "RKO-NATIVE-TYPE-011", "indexing requires a list or constant tuple index")
        if isinstance(node, ast.BinOp) and isinstance(node.op, self._BINOPS):
            left = self.expression(node.left)
            right = self.expression(node.right)
            if isinstance(node.op, ast.Mult) and left.kind == "list" and right == NativeType.int():
                if not isinstance(node.left, ast.List) or len(node.left.elts) != 1:
                    self.error(
                        node,
                        "RKO-NATIVE-EFFECT-008",
                        "V1 list repetition requires a one-element list literal",
                        "write `[scalar] * count`",
                    )
                if left.item is None or left.item.kind in {"list", "tuple"}:
                    self.error(node, "RKO-NATIVE-EFFECT-009", "repetition of nested mutable values is outside V1")
                return left
            if not _is_numeric(left) or not _is_numeric(right):
                self.error(node, "RKO-NATIVE-TYPE-012", "binary arithmetic requires numeric operands")
            if isinstance(node.op, ast.Div):
                return NativeType.float()
            if left == NativeType.float() or right == NativeType.float():
                return NativeType.float()
            return NativeType.int()
        if isinstance(node, ast.UnaryOp):
            if (
                isinstance(node.op, ast.USub)
                and isinstance(node.operand, ast.Constant)
                and isinstance(node.operand.value, int)
                and not isinstance(node.operand.value, bool)
                and node.operand.value == 2**63
            ):
                return NativeType.int()
            operand = self.expression(node.operand)
            if isinstance(node.op, ast.Not):
                self._check(node.operand, NativeType.bool(), operand)
                return NativeType.bool()
            if isinstance(node.op, (ast.UAdd, ast.USub)) and _is_numeric(operand):
                return operand
            self.error(node, "RKO-NATIVE-TYPE-013", "unsupported unary expression")
        if isinstance(node, ast.BoolOp):
            for value in node.values:
                self._check(value, NativeType.bool(), self.expression(value))
            return NativeType.bool()
        if isinstance(node, ast.Compare):
            if any(
                not isinstance(operator, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE))
                for operator in node.ops
            ):
                self.error(node, "RKO-NATIVE-SYNTAX-010", "unsupported V1 comparison operator")
            operands = [node.left, *node.comparators]
            types = [self.expression(operand) for operand in operands]
            for operator, left_type, right_type in zip(node.ops, types, types[1:]):
                if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE)):
                    compatible = (
                        left_type == right_type and _is_numeric(left_type)
                    )
                else:
                    compatible = left_type == right_type
                if not compatible:
                    self.error(node, "RKO-NATIVE-TYPE-020", "comparison operands have incompatible V1 types")
            return NativeType.bool()
        if isinstance(node, ast.IfExp):
            self._check(node.test, NativeType.bool(), self.expression(node.test))
            when_true = self.expression(node.body, expected=expected)
            when_false = self.expression(node.orelse, expected=expected)
            if when_true == when_false:
                return when_true
            self.error(node, "RKO-NATIVE-TYPE-014", "conditional branches have incompatible types")
        if isinstance(node, ast.Call):
            return self._call(node, expected)
        self.error(
            node,
            "RKO-NATIVE-SYNTAX-007",
            f"unsupported V1 expression: {type(node).__name__}",
        )
        raise AssertionError("unreachable")

    def _call(self, node: ast.Call, expected: NativeType | None) -> NativeType:
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "tolist"
            and isinstance(node.func.value, ast.Call)
        ):
            if node.args or node.keywords:
                self.error(node, "RKO-NATIVE-CALL-003", "tolist takes no arguments in V1")
            return self._call(node.func.value, expected)

        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            name = node.func.attr
            if name not in self.signatures:
                self.error(node, "RKO-NATIVE-CALL-004", f"unresolved helper method self.{name}")
            parameter_types, return_type = self.signatures[name]
            if len(node.args) != len(parameter_types) or node.keywords:
                self.error(node, "RKO-NATIVE-CALL-005", f"self.{name} requires {len(parameter_types)} positional arguments")
            for argument, parameter_type in zip(node.args, parameter_types):
                self._check(argument, parameter_type, self.expression(argument, expected=parameter_type))
            return return_type

        canonical = self._canonical_name(node.func)
        if canonical in {"numpy.argsort", "np.argsort"}:
            return self._numpy_argsort(node)

        if isinstance(node.func, ast.Name):
            name = node.func.id
            if name == "len":
                if len(node.args) != 1 or node.keywords:
                    self.error(node, "RKO-NATIVE-CALL-006", "len expects one positional argument")
                value_type = self.expression(node.args[0])
                if value_type.kind not in {"list", "tuple"}:
                    self.error(node, "RKO-NATIVE-TYPE-015", "len requires a list or tuple")
                return NativeType.int()
            if name in {"float", "int", "bool"}:
                if len(node.args) != 1 or node.keywords:
                    self.error(node, "RKO-NATIVE-CALL-007", f"{name} expects one positional argument")
                argument_type = self.expression(node.args[0])
                if argument_type.kind not in {"bool", "int64", "float64"}:
                    self.error(node, "RKO-NATIVE-TYPE-021", f"{name} conversion requires a scalar V1 value")
                return _SCALAR_ANNOTATIONS[name]
            if name in {"min", "max"}:
                if len(node.args) != 2 or node.keywords:
                    self.error(node, "RKO-NATIVE-CALL-008", f"V1 {name} expects exactly two positional arguments")
                left = self.expression(node.args[0])
                right = self.expression(node.args[1])
                if not _is_numeric(left) or not _is_numeric(right):
                    self.error(node, "RKO-NATIVE-TYPE-016", f"{name} requires numeric operands")
                if left != right:
                    self.error(
                        node,
                        "RKO-NATIVE-TYPE-024",
                        f"V1 {name} requires operands with the same concrete type",
                        "convert both operands explicitly to int or float",
                    )
                return left
            if name == "round":
                if not 1 <= len(node.args) <= 2 or node.keywords:
                    self.error(node, "RKO-NATIVE-CALL-009", "round expects one value and optional ndigits")
                value_type = self.expression(node.args[0])
                if value_type != NativeType.float():
                    self.error(node.args[0], "RKO-NATIVE-TYPE-022", "V1 round requires a float value")
                if len(node.args) == 2:
                    self._check(node.args[1], NativeType.int(), self.expression(node.args[1]))
                    if (
                        not isinstance(node.args[1], ast.Constant)
                        or isinstance(node.args[1].value, bool)
                        or not isinstance(node.args[1].value, int)
                    ):
                        self.error(
                            node.args[1],
                            "RKO-NATIVE-CALL-020",
                            "V1 round ndigits must be a constant integer",
                        )
                    if node.args[1].value < 0:
                        self.error(
                            node.args[1],
                            "RKO-NATIVE-CALL-021",
                            "V1 round currently requires non-negative ndigits",
                            "rewrite negative decimal rounding with an explicit domain-specific helper",
                        )
                return NativeType.int() if len(node.args) == 1 else NativeType.float()
            if name == "abs":
                if len(node.args) != 1 or node.keywords:
                    self.error(node, "RKO-NATIVE-CALL-010", "abs expects one positional argument")
                result = self.expression(node.args[0])
                if not _is_numeric(result):
                    self.error(node, "RKO-NATIVE-TYPE-017", "abs requires a numeric argument")
                return result

        if canonical and canonical.startswith("math."):
            if len(node.args) != 1 or node.keywords:
                self.error(node, "RKO-NATIVE-CALL-011", f"{canonical} expects one positional argument in V1")
            argument_type = self.expression(node.args[0])
            if not _is_numeric(argument_type):
                self.error(node.args[0], "RKO-NATIVE-TYPE-023", f"{canonical} requires a numeric value")
            if canonical not in {
                "math.sqrt", "math.exp", "math.log", "math.sin", "math.cos",
                "math.tan", "math.floor", "math.ceil",
            }:
                self.error(node, "RKO-NATIVE-CALL-012", f"unsupported V1 math function {canonical}")
            return NativeType.int() if canonical in {"math.floor", "math.ceil"} else NativeType.float()

        if isinstance(node.func, ast.Attribute):
            owner_type = self.expression(node.func.value)
            method = node.func.attr
            if method == "copy":
                if node.args or node.keywords or owner_type.kind != "list":
                    self.error(node, "RKO-NATIVE-CALL-013", "list.copy takes no arguments")
                if owner_type.item is not None and owner_type.item.kind == "list":
                    self.error(
                        node,
                        "RKO-NATIVE-EFFECT-002",
                        "V1 does not copy nested lists because Python copy() is shallow",
                        "copy nested rows explicitly so aliasing is visible",
                    )
                return owner_type
            if method == "append":
                if len(node.args) != 1 or node.keywords or owner_type.kind != "list" or owner_type.item is None:
                    self.error(node, "RKO-NATIVE-CALL-014", "list.append expects one compatible value")
                appended_type = self.expression(node.args[0], expected=owner_type.item)
                self._check(node.args[0], owner_type.item, appended_type)
                if appended_type.kind == "list":
                    self._check_list_ownership_assignment(node.args[0], appended_type)
                if appended_type.kind == "tuple" and _contains_list(appended_type):
                    self.error(node.args[0], "RKO-NATIVE-EFFECT-015", "appended tuples cannot contain mutable lists")
                return NativeType.none()
            if method == "pop":
                if len(node.args) > 1 or node.keywords or owner_type.kind != "list" or owner_type.item is None:
                    self.error(node, "RKO-NATIVE-CALL-015", "list.pop expects zero or one index")
                if node.args:
                    self._check(node.args[0], NativeType.int(), self.expression(node.args[0]))
                return owner_type.item

        self.error(node, "RKO-NATIVE-CALL-016", f"unsupported or unresolved call: {ast.unparse(node.func)}")
        raise AssertionError("unreachable")

    def _numpy_argsort(self, node: ast.Call) -> NativeType:
        if len(node.args) != 1:
            self.error(node, "RKO-NATIVE-NUMPY-001", "numpy.argsort requires exactly one positional array")
        value_type = self.expression(node.args[0])
        if value_type != NativeType.list(NativeType.float()):
            self.error(node, "RKO-NATIVE-NUMPY-002", "V1 argsort requires list[float]")
        keywords = {keyword.arg: keyword.value for keyword in node.keywords if keyword.arg is not None}
        if set(keywords) != {"kind"}:
            self.error(
                node,
                "RKO-NATIVE-NUMPY-003",
                "V1 argsort requires the explicit keyword kind='stable'",
                "write `np.argsort(values, kind='stable')`",
            )
        kind = keywords["kind"]
        if not isinstance(kind, ast.Constant) or kind.value != "stable":
            self.error(node, "RKO-NATIVE-NUMPY-004", "only kind='stable' is supported by V1 argsort")
        return NativeType.list(NativeType.int())

    def _canonical_name(self, node: ast.expr) -> str | None:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            parent = self._canonical_name(node.value)
            return f"{parent}.{node.attr}" if parent else None
        return None


def _is_numeric(type_info: NativeType) -> bool:
    return type_info.kind in {"int64", "float64"}


class NativeFrontend:
    def __init__(self, env: Any):
        self.env = env
        self.environment_type = type(env)
        source_file_text = inspect.getsourcefile(self.environment_type)
        if source_file_text is None:
            raise NativeCompileError(
                Diagnostic("RKO-NATIVE-SOURCE-001", "the environment class must come from a source file")
            )
        self.source_file = Path(source_file_text).resolve()
        self.filename = str(self.source_file)
        self.source_text = textwrap.dedent(inspect.getsource(self.environment_type))
        self.tree = ast.parse(self.source_text, filename=self.filename, type_comments=True)
        self.class_node = next(
            (node for node in self.tree.body if isinstance(node, ast.ClassDef)),
            None,
        )
        if self.class_node is None:
            raise NativeCompileError(Diagnostic("RKO-NATIVE-SOURCE-002", "unable to parse environment class"))
        self.methods = {
            node.name: node
            for node in self.class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and isinstance(node, ast.FunctionDef)
        }
        self.method_names = set(self.methods)
        self.aliases = import_aliases(self.source_file)

    def build(self) -> EnvironmentIR:
        for root in ("decoder", "cost"):
            if root not in self.methods:
                raise NativeCompileError(
                    Diagnostic("RKO-NATIVE-SOURCE-003", f"environment is missing required method {root}")
                )

        reachable, function_order = self._reachable_methods()
        signatures: dict[str, tuple[list[NativeType], NativeType]] = {}
        parameters_by_method: dict[str, list[ParameterIR]] = {}
        for name in function_order:
            method = self.methods[name]
            self._validate_function_shape(name, method)
            parameters: list[ParameterIR] = []
            for argument in method.args.args:
                if argument.arg == "self":
                    continue
                parameters.append(
                    ParameterIR(argument.arg, parse_annotation(argument.annotation, self.filename))
                )
            return_type = parse_annotation(method.returns, self.filename)
            if any(_contains_none(parameter.type) for parameter in parameters):
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-TYPE-027",
                        f"None is not a valid parameter/container type in {name}",
                        _span(self.filename, method),
                        (name,),
                    )
                )
            if return_type.kind != "none" and _contains_none(return_type):
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-TYPE-028",
                        f"None may only be the complete helper return type in {name}",
                        _span(self.filename, method),
                        (name,),
                    )
                )
            parameters_by_method[name] = parameters
            signatures[name] = ([parameter.type for parameter in parameters], return_type)

        self._validate_root_contract(signatures)

        field_names: set[str] = {"tam_solution"}
        for name in function_order:
            visitor = _SelfFieldVisitor(self.method_names, self.filename, (name,))
            visitor.visit(self.methods[name])
            field_names.update(visitor.fields)

        data_fields: list[DataFieldIR] = []
        field_types: dict[str, NativeType] = {}
        for name in ["tam_solution", *sorted(field_names - {"tam_solution"})]:
            if not hasattr(self.env, name):
                raise NativeCompileError(
                    Diagnostic("RKO-NATIVE-DATA-007", f"environment instance has no field self.{name}")
                )
            raw_value = getattr(self.env, name)
            if callable(raw_value):
                raise NativeCompileError(
                    Diagnostic("RKO-NATIVE-DATA-008", f"self.{name} resolved to a callable, not frozen data")
                )
            type_info = runtime_type(raw_value, path=f"self.{name}")
            field_types[name] = type_info
            data_fields.append(
                DataFieldIR(
                    name,
                    "n" if name == "tam_solution" else f"rko_field_{cpp_identifier(name)}",
                    type_info,
                    normalize_runtime_value(raw_value),
                )
            )

        functions: dict[str, FunctionIR] = {}
        for name in function_order:
            method = self.methods[name]
            frozen_mutation_visitor = _FrozenMutationVisitor(self.filename, name)
            frozen_mutation_visitor.visit(method)
            parameters = parameters_by_method[name]
            mutation_visitor = _DirectMutationVisitor({parameter.name for parameter in parameters})
            mutation_visitor.visit(method)
            for parameter in parameters:
                parameter.mutated = parameter.name in mutation_visitor.mutated
            dependency_visitor = _MethodDependencyVisitor(self.method_names)
            dependency_visitor.visit(method)
            typer = _FunctionTyper(
                filename=self.filename,
                function_name=name,
                parameters=parameters,
                return_type=signatures[name][1],
                fields=field_types,
                signatures=signatures,
                method_names=self.method_names,
                aliases=self.aliases,
            )
            local_types, loop_targets, empty_list_types = typer.analyze(method)
            if signatures[name][1] != NativeType.none() and not self._block_always_returns(method.body):
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-CONTROL-001",
                        f"reachable function {name} does not return a value on every V1 control-flow path",
                        _span(self.filename, method),
                        (name,),
                        "add an explicit return after conditional and loop paths",
                    )
                )
            functions[name] = FunctionIR(
                name=name,
                cpp_name=f"RkoFn_{cpp_identifier(name)}",
                node=method,
                parameters=parameters,
                return_type=signatures[name][1],
                local_types=local_types,
                loop_targets=loop_targets,
                empty_list_types=empty_list_types,
                dependencies=dependency_visitor.dependencies,
                source_line=method.lineno,
            )

        self._propagate_mutations(functions)
        self._validate_expression_effects(functions)
        self._validate_mutation_effects(functions)
        self._validate_iteration_effects(functions)
        tam_solution = getattr(self.env, "tam_solution")
        if (
            not isinstance(tam_solution, int)
            or isinstance(tam_solution, bool)
            or tam_solution <= 0
            or tam_solution > 2**31 - 1
        ):
            raise NativeCompileError(
                Diagnostic("RKO-NATIVE-DATA-009", "tam_solution must be a positive 32-bit int")
            )
        return EnvironmentIR(
            class_name=self.environment_type.__qualname__,
            source_file=self.filename,
            source_text=self.source_text,
            functions=functions,
            function_order=function_order,
            data_fields=data_fields,
            import_aliases=self.aliases,
            tam_solution=tam_solution,
        )

    @staticmethod
    def _block_always_returns(statements: list[ast.stmt]) -> bool:
        for statement in statements:
            if isinstance(statement, ast.Return):
                return True
            if (
                isinstance(statement, ast.If)
                and statement.orelse
                and NativeFrontend._block_always_returns(statement.body)
                and NativeFrontend._block_always_returns(statement.orelse)
            ):
                return True
        return False

    def _validate_function_shape(self, name: str, method: ast.FunctionDef) -> None:
        arguments = method.args
        if (
            arguments.posonlyargs
            or arguments.vararg is not None
            or arguments.kwonlyargs
            or arguments.kwarg is not None
        ):
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-SOURCE-004",
                    f"{name} must use fixed positional-or-keyword parameters in V1",
                    _span(self.filename, method),
                    (name,),
                    "remove positional-only, variadic and keyword-only parameters from the native hot path",
                )
            )
        if not arguments.args or arguments.args[0].arg != "self":
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-SOURCE-005",
                    f"{name} must be an instance method whose first parameter is self",
                    _span(self.filename, method),
                    (name,),
                )
            )
        if method.decorator_list:
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-SOURCE-006",
                    f"decorators on reachable method {name} are outside V1",
                    _span(self.filename, method.decorator_list[0]),
                    (name,),
                )
            )

    def _validate_root_contract(
        self, signatures: dict[str, tuple[list[NativeType], NativeType]]
    ) -> None:
        decoder_parameters, decoder_return = signatures["decoder"]
        expected_keys = NativeType.list(NativeType.float())
        if decoder_parameters != [expected_keys] or decoder_return.kind != "list":
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-CONTRACT-001",
                    "decoder must have signature `(self, keys: list[float]) -> list[T]`",
                    _span(self.filename, self.methods["decoder"]),
                    ("decoder",),
                )
            )

        cost_parameters, cost_return = signatures["cost"]
        if (
            len(cost_parameters) != 2
            or cost_parameters[0] != decoder_return
            or cost_parameters[1] != NativeType.bool()
            or cost_return != NativeType.float()
        ):
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-CONTRACT-002",
                    "cost must have signature `(self, solution: decoder_return, final_solution: bool = False) -> float`",
                    _span(self.filename, self.methods["cost"]),
                    ("cost",),
                )
            )

        defaults = self.methods["cost"].args.defaults
        if (
            not defaults
            or not isinstance(defaults[-1], ast.Constant)
            or defaults[-1].value is not False
        ):
            raise NativeCompileError(
                Diagnostic(
                    "RKO-NATIVE-CONTRACT-003",
                    "cost final_solution must have the explicit default False",
                    _span(self.filename, self.methods["cost"]),
                    ("cost",),
                )
            )

    def _validate_mutation_effects(self, functions: dict[str, FunctionIR]) -> None:
        read_only_roots = (("decoder", 0, "keys"), ("cost", 0, "solution"))
        for function_name, parameter_index, contract_name in read_only_roots:
            function = functions[function_name]
            if function.parameters[parameter_index].mutated:
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-EFFECT-010",
                        f"root parameter {contract_name} is read-only in V1",
                        _span(self.filename, function.node),
                        (function_name,),
                        "copy it to an annotated local list before mutation",
                    )
                )

        for caller in functions.values():
            for node in ast.walk(caller.node):
                if not (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr in functions
                ):
                    continue
                callee = functions[node.func.attr]
                for argument, parameter in zip(node.args, callee.parameters):
                    if not parameter.mutated:
                        continue
                    if _is_frozen_self_expression(argument):
                        raise NativeCompileError(
                            Diagnostic(
                                "RKO-NATIVE-EFFECT-011",
                                f"self.{node.func.attr} mutates an argument backed by frozen environment data",
                                _span(self.filename, argument),
                                (caller.name, node.func.attr),
                                "make an explicit flat local copy before calling the helper",
                            )
                        )
                    if not isinstance(argument, (ast.Name, ast.Subscript)):
                        raise NativeCompileError(
                            Diagnostic(
                                "RKO-NATIVE-EFFECT-012",
                                f"mutable parameter {parameter.name} of self.{node.func.attr} requires an owned local lvalue",
                                _span(self.filename, argument),
                                (caller.name, node.func.attr),
                                "assign the value to an annotated local list before the call",
                            )
                        )

    def _validate_expression_effects(self, functions: dict[str, FunctionIR]) -> None:
        for function in functions.values():
            parents: dict[ast.AST, ast.AST] = {}
            for parent in ast.walk(function.node):
                for child in ast.iter_child_nodes(parent):
                    parents[child] = parent

            for node in ast.walk(function.node):
                if not isinstance(node, ast.Call):
                    continue
                effectful = (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr in {"append", "pop"}
                )
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "self"
                    and node.func.attr in functions
                ):
                    effectful = effectful or any(
                        parameter.mutated
                        for parameter in functions[node.func.attr].parameters
                    )
                if not effectful:
                    continue

                parent = parents.get(node)
                is_simple_root = (
                    isinstance(parent, ast.Expr) and parent.value is node
                ) or (
                    isinstance(parent, ast.Assign) and parent.value is node
                ) or (
                    isinstance(parent, ast.AnnAssign) and parent.value is node
                ) or (
                    isinstance(parent, ast.Return) and parent.value is node
                )
                if not is_simple_root:
                    raise NativeCompileError(
                        Diagnostic(
                            "RKO-NATIVE-EFFECT-017",
                            "effectful pop/helper calls must be a complete statement, assignment RHS or return expression",
                            _span(self.filename, node),
                            (function.name,),
                            "split the expression into ordered local assignments",
                        )
                    )

    def _validate_iteration_effects(self, functions: dict[str, FunctionIR]) -> None:
        for function in functions.values():
            for loop in ast.walk(function.node):
                if not isinstance(loop, ast.For):
                    continue
                iterable: ast.expr = loop.iter
                if (
                    isinstance(iterable, ast.Call)
                    and isinstance(iterable.func, ast.Name)
                    and iterable.func.id == "range"
                ):
                    continue
                if (
                    isinstance(iterable, ast.Call)
                    and isinstance(iterable.func, ast.Name)
                    and iterable.func.id == "enumerate"
                ):
                    iterable = iterable.args[0]
                iterable_root = _root_name(iterable)
                if iterable_root is None:
                    continue

                mutated_roots: set[str] = set()
                for statement in loop.body:
                    for node in ast.walk(statement):
                        if isinstance(node, (ast.Assign, ast.AugAssign)):
                            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                            for target in targets:
                                if isinstance(target, ast.Subscript):
                                    root = _root_name(target)
                                    if root is not None:
                                        mutated_roots.add(root)
                        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                            if node.func.attr in {"append", "pop", "clear"}:
                                root = _root_name(node.func.value)
                                if root is not None:
                                    mutated_roots.add(root)
                            if (
                                isinstance(node.func.value, ast.Name)
                                and node.func.value.id == "self"
                                and node.func.attr in functions
                            ):
                                callee = functions[node.func.attr]
                                for argument, parameter in zip(node.args, callee.parameters):
                                    if parameter.mutated:
                                        root = _root_name(argument)
                                        if root is not None:
                                            mutated_roots.add(root)

                if iterable_root in mutated_roots:
                    raise NativeCompileError(
                        Diagnostic(
                            "RKO-NATIVE-EFFECT-014",
                            f"list {iterable_root} is mutated while it is being iterated",
                            _span(self.filename, loop),
                            (function.name,),
                            "iterate over an explicit flat copy or use an index-based loop",
                        )
                    )

    def _reachable_methods(self) -> tuple[set[str], list[str]]:
        reachable: set[str] = set()
        postorder: list[str] = []
        active: set[str] = set()

        def visit(name: str) -> None:
            if name in reachable:
                return
            if name in active:
                raise NativeCompileError(
                    Diagnostic(
                        "RKO-NATIVE-CALL-017",
                        f"recursive helper cycle involving {name} is outside V1",
                    )
                )
            active.add(name)
            visitor = _MethodDependencyVisitor(self.method_names)
            visitor.visit(self.methods[name])
            for dependency in visitor.dependencies:
                visit(dependency)
            active.remove(name)
            reachable.add(name)
            postorder.append(name)

        visit("decoder")
        visit("cost")
        return reachable, postorder

    @staticmethod
    def _propagate_mutations(functions: dict[str, FunctionIR]) -> None:
        changed = True
        while changed:
            changed = False
            for function in functions.values():
                parameter_by_name = {parameter.name: parameter for parameter in function.parameters}
                for node in ast.walk(function.node):
                    if not (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "self"
                        and node.func.attr in functions
                    ):
                        continue
                    callee = functions[node.func.attr]
                    for argument, callee_parameter in zip(node.args, callee.parameters):
                        if not callee_parameter.mutated:
                            continue
                        root = _root_name(argument)
                        if root in parameter_by_name and not parameter_by_name[root].mutated:
                            parameter_by_name[root].mutated = True
                            changed = True


def build_environment_ir(env: Any) -> EnvironmentIR:
    return NativeFrontend(env).build()
