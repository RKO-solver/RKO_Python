"""High-level Python -> native C++ RKO -> Python execution pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

from .gera_program import compile_environment
from .model import CompilationArtifact
from .build_make import (
    RKOConfig,
    RKOResult,
    ToolchainBuild,
    build_toolchain,
    run_evaluator,
    run_rko,
)


class NativeValidationError(RuntimeError):
    """Raised when the native/Python handoff cannot be proven equivalent."""


@dataclass(frozen=True)
class NativeOptimizationResult:
    """A complete optimization result reconstructed in the Python runtime."""

    artifact: CompilationArtifact
    build: ToolchainBuild
    native_result: RKOResult
    keys: tuple[float, ...]
    solution: Any
    python_cost: float
    native_cost: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "keys": list(self.keys),
            "solution": self.solution,
            "python_cost": self.python_cost,
            "native_cost": self.native_cost,
            "native_result": self.native_result.as_dict(),
            "generated_directory": str(self.artifact.output_directory),
            "build_directory": str(self.build.prepared.build_dir),
        }


def optimize_environment(
    env: Any,
    work_directory: str | Path,
    *,
    max_time_seconds: int = 1,
    config: RKOConfig | None = None,
    cpp_program_source: str | Path | None = None,
    cxx: str = "g++",
    require_exact_keys: bool = True,
    relative_tolerance: float = 1.0e-12,
    absolute_tolerance: float = 1.0e-9,
    stream_stdout: bool = False,
) -> NativeOptimizationResult:
    """Compile an initialized V1 env, run RKO C++, and return its Python solution.

    The original environment reader and ``__init__`` have already run when this
    function is called.  Only its reachable V1 hot path and frozen data snapshot
    cross into C++.  Returned keys are evaluated once more in both runtimes
    before the result is accepted.
    """

    root = Path(work_directory).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    artifact = compile_environment(env, root / "generated")
    build = build_toolchain(
        artifact.output_directory,
        root / "build",
        cpp_program_source=cpp_program_source,
        config=config,
        cxx=cxx,
    )
    native_result = run_rko(build, max_time_seconds=max_time_seconds, stream_stdout=stream_stdout)

    if require_exact_keys and native_result.output_is_lossy:
        raise NativeValidationError(
            "The selected C++ RKO only exposed rounded textual keys; exact Python solution reconstruction is unsafe. "
            "Use the RKO_RESULT_V1 output hook or set require_exact_keys=False explicitly."
        )
    if len(native_result.keys) != artifact.environment_ir.tam_solution:
        raise NativeValidationError(
            "RKO returned {} keys, expected {}".format(
                len(native_result.keys), artifact.environment_ir.tam_solution
            )
        )

    keys = tuple(native_result.keys)
    solution = env.decoder(list(keys))
    python_cost = float(env.cost(solution, False))
    native_cost = run_evaluator(build, keys).cost
    if not math.isfinite(python_cost) or not math.isfinite(native_cost):
        raise NativeValidationError("V1 requires finite Python and native objective values")
    if not math.isclose(
        python_cost,
        native_cost,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise NativeValidationError(
            "Returned-key differential failed: Python={!r}, C++={!r}".format(
                python_cost, native_cost
            )
        )
    if not native_result.output_is_lossy and not math.isclose(
        native_result.cost,
        native_cost,
        rel_tol=relative_tolerance,
        abs_tol=absolute_tolerance,
    ):
        raise NativeValidationError(
            "RKO machine result does not reproduce under the native evaluator: optimized={!r}, reevaluated={!r}".format(
                native_result.cost, native_cost
            )
        )

    return NativeOptimizationResult(
        artifact=artifact,
        build=build,
        native_result=native_result,
        keys=keys,
        solution=solution,
        python_cost=python_cost,
        native_cost=native_cost,
    )
