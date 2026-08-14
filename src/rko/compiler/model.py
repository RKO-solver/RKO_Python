"""Core immutable model for the RKO native language V1 compiler."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


LANGUAGE_VERSION = "rko-native-v1"


@dataclass(frozen=True)
class NativeType:
    kind: str
    item: "NativeType | None" = None
    items: tuple["NativeType", ...] = ()

    @staticmethod
    def bool() -> "NativeType":
        return NativeType("bool")

    @staticmethod
    def int() -> "NativeType":
        return NativeType("int64")

    @staticmethod
    def float() -> "NativeType":
        return NativeType("float64")

    @staticmethod
    def none() -> "NativeType":
        return NativeType("none")

    @staticmethod
    def list(item: "NativeType") -> "NativeType":
        return NativeType("list", item=item)

    @staticmethod
    def tuple(items: tuple["NativeType", ...]) -> "NativeType":
        return NativeType("tuple", items=items)

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {"kind": self.kind}
        if self.item is not None:
            result["item"] = self.item.to_data()
        if self.items:
            result["items"] = [item.to_data() for item in self.items]
        return result

    def display(self) -> str:
        if self.kind == "list" and self.item is not None:
            return f"list[{self.item.display()}]"
        if self.kind == "tuple":
            return f"tuple[{', '.join(item.display() for item in self.items)}]"
        return self.kind


@dataclass(frozen=True)
class SourceSpan:
    filename: str
    line: int
    column: int = 0
    end_line: int | None = None
    end_column: int | None = None

    def to_data(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "line": self.line,
            "column": self.column,
            "end_line": self.end_line,
            "end_column": self.end_column,
        }


@dataclass(frozen=True)
class Diagnostic:
    code: str
    message: str
    span: SourceSpan | None = None
    call_chain: tuple[str, ...] = ()
    adaptation: str | None = None

    def to_data(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "call_chain": list(self.call_chain),
        }
        if self.span is not None:
            result["span"] = self.span.to_data()
        if self.adaptation is not None:
            result["adaptation"] = self.adaptation
        return result


class NativeCompileError(ValueError):
    """Raised with a structured diagnostic for unsupported V1 source."""

    def __init__(self, diagnostic: Diagnostic):
        self.diagnostic = diagnostic
        location = ""
        if diagnostic.span is not None:
            location = f"{diagnostic.span.filename}:{diagnostic.span.line}: "
        chain = ""
        if diagnostic.call_chain:
            chain = f" [call chain: {' -> '.join(diagnostic.call_chain)}]"
        adaptation = ""
        if diagnostic.adaptation:
            adaptation = f" Adaptation: {diagnostic.adaptation}"
        super().__init__(
            f"{diagnostic.code}: {location}{diagnostic.message}{chain}.{adaptation}"
        )


@dataclass
class ParameterIR:
    name: str
    type: NativeType
    mutated: bool = False

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.to_data(),
            "mutated": self.mutated,
        }


@dataclass
class FunctionIR:
    name: str
    cpp_name: str
    node: Any
    parameters: list[ParameterIR]
    return_type: NativeType
    local_types: dict[str, NativeType] = field(default_factory=dict)
    loop_targets: set[str] = field(default_factory=set)
    empty_list_types: dict[int, NativeType] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    source_line: int = 0

    def to_data(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "cpp_name": self.cpp_name,
            "parameters": [parameter.to_data() for parameter in self.parameters],
            "return_type": self.return_type.to_data(),
            "local_types": {
                name: type_info.to_data()
                for name, type_info in sorted(self.local_types.items())
            },
            "dependencies": self.dependencies,
            "source_line": self.source_line,
        }


@dataclass
class DataFieldIR:
    name: str
    cpp_name: str
    type: NativeType
    value: Any

    def to_data(self, include_value: bool = True) -> dict[str, Any]:
        result = {
            "name": self.name,
            "cpp_name": self.cpp_name,
            "type": self.type.to_data(),
        }
        if include_value:
            result["value"] = self.value
        return result


@dataclass
class EnvironmentIR:
    class_name: str
    source_file: str
    source_text: str
    functions: dict[str, FunctionIR]
    function_order: list[str]
    data_fields: list[DataFieldIR]
    import_aliases: dict[str, str]
    tam_solution: int

    def to_data(self) -> dict[str, Any]:
        return {
            "language_version": LANGUAGE_VERSION,
            "class_name": self.class_name,
            "source_file": self.source_file,
            "tam_solution": self.tam_solution,
            "import_aliases": self.import_aliases,
            "function_order": self.function_order,
            "functions": {
                name: function.to_data()
                for name, function in sorted(self.functions.items())
            },
            "data_fields": [field.to_data() for field in self.data_fields],
        }


@dataclass(frozen=True)
class CompilationArtifact:
    output_directory: Path
    problem_header: Path
    instance_data: Path
    schema: Path
    typed_ir: Path
    report: Path
    environment_ir: EnvironmentIR
