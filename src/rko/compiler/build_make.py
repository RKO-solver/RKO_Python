"""WSL toolchain for generated native RKO environments.

The compiler front-end writes ``Problem.h`` and ``instance.rko-data``.  This
module deliberately knows nothing about the Python AST or the generated IR: it
only creates an isolated copy of the original C++ Program, builds it in WSL,
and runs either a single-solution evaluator or the complete RKO executable.

The legacy human ``Output.h`` report prints random keys with three decimal
places and the objective with five.  The bundled C++ RKO additionally emits a
versioned ``RKO_RESULT_V1`` record with ``max_digits10`` precision.  The parser
prefers that exact protocol and retains a deliberately lossy fallback for
unmodified external copies of the original RKO.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, replace
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Sequence, Tuple, Union


PathLike = Union[str, os.PathLike]

_SUPPORTED_ALGORITHMS = (
    "BRKGA",
    "SA",
    "GRASP",
    "ILS",
    "VNS",
    "PSO",
    "GA",
    "LNS",
    "BRKGA-CS",
    "MultiStart",
    "IPR",
)

_FLOAT_TOKEN = (
    r"[+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    r"|inf(?:inity)?|nan)"
)
_ANSI_ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))")
_EVALUATOR_MARKER = "RKO_EVALUATOR_RESULT"
_RKO_RESULT_MARKER = "@@RKO_RESULT_V1@@"
_BUILD_OWNERSHIP_MARKER = ".rko-native-build-v1"


class ToolchainError(RuntimeError):
    """Base class for failures in the native RKO toolchain."""


class WSLUnavailableError(ToolchainError):
    """Raised when WSL (or the requested command inside WSL) is unavailable."""


class ToolchainCommandError(ToolchainError):
    """A command executed by the toolchain returned a non-zero status."""

    def __init__(
        self,
        message: str,
        *,
        command: Sequence[str],
        returncode: Optional[int],
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        details = [message, "command: " + shlex.join(list(command))]
        if returncode is not None:
            details.append("exit status: {}".format(returncode))
        if stdout.strip():
            details.append("stdout:\n{}".format(stdout.rstrip()))
        if stderr.strip():
            details.append("stderr:\n{}".format(stderr.rstrip()))
        super().__init__("\n".join(details))
        self.command = tuple(command)
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class RKOConfig:
    """Configuration written into the isolated C++ Program copy.

    ``MultiStart`` is the conservative default for V1: it exercises the
    original RKO machinery without requiring every decoder to support the more
    elaborate neighbourhood assumptions made by all metaheuristics.
    """

    algorithms: Tuple[str, ...] = ("MultiStart",)
    max_runs: int = 1
    debug: bool = True
    control: int = 0
    strategy: int = 1
    restart: float = 1.0
    pool_size: int = 10

    def render(self) -> str:
        if not isinstance(self.algorithms, tuple) or any(
            not isinstance(name, str) for name in self.algorithms
        ):
            raise TypeError("algorithms must be a tuple of RKO algorithm names")
        unknown = [name for name in self.algorithms if name not in _SUPPORTED_ALGORITHMS]
        if unknown:
            raise ValueError("Unknown RKO algorithm(s): {}".format(", ".join(unknown)))
        if not self.algorithms:
            raise ValueError("At least one RKO algorithm must be configured")
        if len(self.algorithms) > 100:
            raise ValueError("The original RKO supports at most 100 configured algorithms")
        if len(set(self.algorithms)) != len(self.algorithms):
            raise ValueError("RKO algorithms must not be repeated in one configuration")
        for field_name, value in (
            ("max_runs", self.max_runs),
            ("control", self.control),
            ("strategy", self.strategy),
            ("pool_size", self.pool_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError("{} must be an integer".format(field_name))
        if not isinstance(self.debug, bool):
            raise TypeError("debug must be bool")
        if self.max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        if self.control not in (0, 1):
            raise ValueError("control must be 0 or 1")
        if self.strategy not in (1, 2):
            raise ValueError("strategy must be 1 or 2")
        if (
            isinstance(self.restart, bool)
            or not isinstance(self.restart, (int, float))
            or not math.isfinite(self.restart)
            or not 0.0 < self.restart <= 1.0
        ):
            raise ValueError("restart must be a finite fraction in (0, 1]")
        if self.pool_size < 1:
            raise ValueError("pool_size must be at least 1")

        lines = list(self.algorithms)
        lines.extend(
            (
                "",
                "MAXRUNS {}".format(self.max_runs),
                "debug {}".format(1 if self.debug else 0),
                "control {}".format(self.control),
                "strategy {}".format(self.strategy),
                "restart {:.17g}".format(self.restart),
                "sizePool {}".format(self.pool_size),
                "",
            )
        )
        return "\n".join(lines)


@dataclass(frozen=True)
class PreparedProgram:
    """Filesystem layout prepared for compilation."""

    output_dir: Path
    build_dir: Path
    program_dir: Path
    problem_header: Path
    instance_file: Path
    config: RKOConfig
    machine_protocol_supported: bool


@dataclass(frozen=True)
class ToolchainBuild:
    """The two native executables built for one generated environment."""

    prepared: PreparedProgram
    evaluator_executable: Path
    rko_executable: Path


@dataclass(frozen=True)
class EvaluatorResult:
    """Exact objective value returned for a caller-supplied key vector."""

    cost: float
    stdout: str
    stderr: str


@dataclass(frozen=True)
class RKOResult:
    """Parsed result from the original RKO command-line program.

    Exact bundled builds use the machine-readable V1 protocol.  External
    legacy builds fall back to the precision described by
    ``keys_decimal_places`` and ``cost_decimal_places``.  Complete output is
    retained for archival and diagnostics.
    """

    keys: List[float]
    cost: float
    time_to_best: Optional[float]
    total_time: Optional[float]
    best_metaheuristic: Optional[str]
    instance: Optional[str]
    stdout: str
    stderr: str = ""
    keys_decimal_places: Optional[int] = 3
    cost_decimal_places: Optional[int] = 5
    output_is_lossy: bool = True
    precision_warning: Optional[str] = (
        "The unmodified C++ RKO Output.h prints keys with 3 decimal places "
        "and objective values with 5 decimal places. Re-evaluate the returned "
        "keys before treating either value as exact."
    )
    protocol_version: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        """Return the result in the dictionary shape used by older studies."""

        return {
            "keys": list(self.keys),
            "cost": self.cost,
            "time_to_best": self.time_to_best,
            "total_time": self.total_time,
            "best_metaheuristic": self.best_metaheuristic,
            "instance": self.instance,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "keys_decimal_places": self.keys_decimal_places,
            "cost_decimal_places": self.cost_decimal_places,
            "output_is_lossy": self.output_is_lossy,
            "precision_warning": self.precision_warning,
            "protocol_version": self.protocol_version,
        }


def default_cpp_program_source() -> Path:
    """Return the bundled original C++ ``Program`` directory."""

    package_dir = Path(__file__).resolve().parent / "cpp_template"
    if (package_dir / "src" / "Main" / "main.cpp").is_file():
        return package_dir
    repository_root = Path(__file__).resolve().parents[3]
    return repository_root / "RKO_C++" / "RKO_Cpp_v2.0-main" / "Program"


def prepare_program(
    output_dir: PathLike,
    build_dir: PathLike,
    *,
    cpp_program_source: Optional[PathLike] = None,
    config: Optional[RKOConfig] = None,
) -> PreparedProgram:
    """Create an isolated C++ Program populated with generated artifacts.

    The source ``Program`` is copied rather than edited in place.  Only the
    generated ``Problem.h`` and the isolated configuration are replaced; in
    particular, ``Output.h`` remains byte-for-byte identical to the upstream
    source.
    """

    generated = Path(output_dir).expanduser().resolve()
    destination_root = Path(build_dir).expanduser().resolve()
    source_program = Path(
        cpp_program_source if cpp_program_source is not None else default_cpp_program_source()
    ).expanduser().resolve()

    generated_problem = generated / "Problem.h"
    generated_instance = generated / "instance.rko-data"
    for required in (generated_problem, generated_instance):
        if not required.is_file():
            raise FileNotFoundError("Generated compiler artifact not found: {}".format(required))
    if not source_program.is_dir():
        raise FileNotFoundError("Original C++ RKO Program not found: {}".format(source_program))

    program_dir = destination_root / "Program"
    if _paths_overlap(source_program, program_dir):
        raise ToolchainError(
            "The isolated build Program must not contain, or be contained by, "
            "the original Program: {}".format(program_dir)
        )
    if _path_is_within(generated, program_dir):
        raise ToolchainError(
            "The generated artifact directory must be outside the isolated "
            "Program destination: {}".format(program_dir)
        )

    chosen_config = config if config is not None else RKOConfig()
    # Validate before touching an existing build directory.
    chosen_config.render()
    destination_root.mkdir(parents=True, exist_ok=True)
    ownership_marker = program_dir / _BUILD_OWNERSHIP_MARKER
    if program_dir.exists():
        if program_dir.is_symlink():
            raise ToolchainError("Refusing to replace symlinked build Program: {}".format(program_dir))
        if not ownership_marker.is_file() or ownership_marker.read_text(encoding="utf-8").strip() != _BUILD_OWNERSHIP_MARKER:
            raise ToolchainError(
                "Refusing to replace an existing Program without the RKO Native V1 ownership marker: {}".format(
                    program_dir
                )
            )
        shutil.rmtree(str(program_dir))
    shutil.copytree(str(source_program), str(program_dir))
    ownership_marker.write_text(_BUILD_OWNERSHIP_MARKER + "\n", encoding="utf-8")

    problem_header = program_dir / "src" / "Problem" / "Problem.h"
    shutil.copy2(str(generated_problem), str(problem_header))
    instance_file = destination_root / "instance.rko-data"
    shutil.copy2(str(generated_instance), str(instance_file))

    config_path = program_dir / "config" / "config_tests.conf"
    # The upstream parser removes ``\n`` but not ``\r`` before comparing
    # algorithm names.  Write bytes so Windows cannot translate LF to CRLF.
    config_path.write_bytes(chosen_config.render().encode("utf-8"))
    # Non-debug mode writes to ../Results relative to Program.
    (destination_root / "Results").mkdir(exist_ok=True)

    output_source = program_dir / "src" / "Output.h"
    main_source = program_dir / "src" / "Main" / "main.cpp"
    machine_protocol_supported = (
        _RKO_RESULT_MARKER in output_source.read_text(encoding="utf-8", errors="replace")
        and "WriteMachineReadableResult(bestSolution)" in main_source.read_text(encoding="utf-8", errors="replace")
    )
    if not chosen_config.debug and not machine_protocol_supported:
        raise ToolchainError(
            "debug=False requires an RKO C++ source with the RKO_RESULT_V1 machine-readable output hook"
        )

    return PreparedProgram(
        output_dir=generated,
        build_dir=destination_root,
        program_dir=program_dir,
        problem_header=problem_header,
        instance_file=instance_file,
        config=chosen_config,
        machine_protocol_supported=machine_protocol_supported,
    )


def compile_evaluator(
    prepared: PreparedProgram,
    *,
    cxx: str = "g++",
    timeout: Optional[float] = 120.0,
) -> Path:
    """Build the exact, single-key-vector evaluator in WSL."""

    source = prepared.build_dir / "evaluate_generated_problem.cpp"
    source.write_text(_evaluator_source(), encoding="utf-8")
    executable = prepared.build_dir / "evaluate_generated_problem"
    completed = _run_in_wsl(
        (
            cxx,
            "-std=c++20",
            "-O2",
            source,
            "-o",
            executable,
        ),
        cwd=prepared.build_dir,
        timeout=timeout,
    )
    _write_command_logs(prepared.build_dir, "evaluator-build", completed)
    return executable


def compile_rko(
    prepared: PreparedProgram,
    *,
    cxx: str = "g++",
    timeout: Optional[float] = 180.0,
) -> Path:
    """Build the isolated original RKO program (plus its output hook) in WSL."""

    executable = prepared.program_dir / "runRKO"
    completed = _run_in_wsl(
        (
            cxx,
            "-std=c++20",
            "-O2",
            "-fopenmp",
            "src/Main/main.cpp",
            "-o",
            executable,
        ),
        cwd=prepared.program_dir,
        timeout=timeout,
    )
    _write_command_logs(prepared.build_dir, "rko-build", completed)
    return executable


def build_toolchain(
    output_dir: PathLike,
    build_dir: PathLike,
    *,
    cpp_program_source: Optional[PathLike] = None,
    config: Optional[RKOConfig] = None,
    cxx: str = "g++",
) -> ToolchainBuild:
    """Prepare and build both generated-environment executables."""

    prepared = prepare_program(
        output_dir,
        build_dir,
        cpp_program_source=cpp_program_source,
        config=config,
    )
    evaluator = compile_evaluator(prepared, cxx=cxx)
    rko = compile_rko(prepared, cxx=cxx)
    return ToolchainBuild(
        prepared=prepared,
        evaluator_executable=evaluator,
        rko_executable=rko,
    )


def run_evaluator(
    build: ToolchainBuild,
    keys: Sequence[float],
    *,
    timeout: Optional[float] = 60.0,
) -> EvaluatorResult:
    """Evaluate one random-key vector with full ``double`` output precision."""

    key_arguments = [_format_key(value) for value in keys]
    completed = _run_in_wsl(
        (build.evaluator_executable, build.prepared.instance_file),
        cwd=build.prepared.build_dir,
        timeout=timeout,
        input_text="\n".join(key_arguments) + ("\n" if key_arguments else ""),
    )
    matches = re.findall(
        r"(?im)^\s*{}\s+({})\s*$".format(re.escape(_EVALUATOR_MARKER), _FLOAT_TOKEN),
        _strip_ansi(completed.stdout),
    )
    if not matches:
        raise ToolchainError(
            "The generated evaluator completed without a parseable result marker.\n"
            "stdout:\n{}\nstderr:\n{}".format(completed.stdout, completed.stderr)
        )
    cost = float(matches[-1])
    if not math.isfinite(cost):
        raise ToolchainError("The generated evaluator returned a non-finite V1 objective: {!r}".format(cost))
    return EvaluatorResult(
        cost=cost,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def run_rko(
    build: ToolchainBuild,
    *,
    max_time_seconds: int = 1,
    timeout: Optional[float] = None,
    stream_stdout: bool = False,
) -> RKOResult:
    """Run and parse the complete original RKO executable in WSL."""

    if isinstance(max_time_seconds, bool) or not isinstance(max_time_seconds, int):
        raise TypeError("max_time_seconds must be an integer")
    if max_time_seconds < 1:
        raise ValueError("max_time_seconds must be at least 1")
    effective_timeout = timeout
    if effective_timeout is None:
        expected_runtime = float(max_time_seconds * build.prepared.config.max_runs)
        effective_timeout = max(60.0, expected_runtime + 30.0)
    completed = _run_in_wsl(
        (
            build.rko_executable,
            build.prepared.instance_file,
            str(max_time_seconds),
        ),
        cwd=build.prepared.program_dir,
        timeout=effective_timeout,
        stream_stdout=stream_stdout,
    )
    parsed = parse_rko_stdout(completed.stdout)
    return replace(parsed, stdout=completed.stdout, stderr=completed.stderr)


def parse_rko_stdout(stdout: str) -> RKOResult:
    """Parse the final solution block printed by the original ``Output.h``.

    Debug output can contain arbitrary lines before the final report, so the
    parser searches for complete ``sol``/``ofv`` pairs and selects the last
    pair.  Scientific notation, infinities, CRLF, and ANSI terminal escapes are
    accepted.
    """

    cleaned = _strip_ansi(stdout).replace("\r\n", "\n").replace("\r", "\n")
    marker_lines: List[str] = []
    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped and stripped.split(maxsplit=1)[0] == _RKO_RESULT_MARKER:
            marker_lines.append(stripped)
    if marker_lines:
        marker_line = marker_lines[-1]
        tokens = marker_line.split()
        if not tokens or tokens[0] != _RKO_RESULT_MARKER:
            raise ToolchainError("Malformed RKO_RESULT_V1 marker line: {!r}".format(marker_line))
        if len(tokens) < 3 or re.fullmatch(r"\d+", tokens[1]) is None:
            raise ToolchainError("Malformed RKO_RESULT_V1 header: {!r}".format(marker_line))
        dimension = int(tokens[1])
        if dimension > 10_000_000:
            raise ToolchainError("RKO_RESULT_V1 dimension is unreasonably large: {}".format(dimension))
        if len(tokens) != dimension + 3:
            raise ToolchainError(
                "RKO_RESULT_V1 expected {} tokens, received {}".format(dimension + 3, len(tokens))
            )
        numeric_tokens = tokens[2:]
        for position, token in enumerate(numeric_tokens):
            if re.fullmatch(_FLOAT_TOKEN, token, flags=re.IGNORECASE) is None:
                raise ToolchainError(
                    "Invalid RKO_RESULT_V1 numeric token at position {}: {!r}".format(position, token)
                )
        numeric_values = [float(token) for token in numeric_tokens]
        if any(not math.isfinite(value) for value in numeric_values):
            raise ToolchainError("RKO_RESULT_V1 requires finite objective and key values")
        return RKOResult(
            keys=numeric_values[1:],
            cost=numeric_values[0],
            time_to_best=_last_labeled_float(cleaned, "Best time"),
            total_time=_last_labeled_float(cleaned, "Total time"),
            best_metaheuristic=_last_labeled_text(cleaned, "Best MH"),
            instance=_last_labeled_text(cleaned, "Instance"),
            stdout=stdout,
            keys_decimal_places=None,
            cost_decimal_places=None,
            output_is_lossy=False,
            precision_warning=None,
            protocol_version="RKO_RESULT_V1",
        )

    pair_pattern = re.compile(
        r"(?ims)^[ \t]*sol[ \t]*:[ \t]*(.*?)\n[ \t]*ofv[ \t]*:[ \t]*({})(?=\s|$)".format(
            _FLOAT_TOKEN
        )
    )
    pairs = list(pair_pattern.finditer(cleaned))
    if not pairs:
        raise ToolchainError(
            "Unable to find a complete 'sol:'/'ofv:' result block in RKO stdout:\n{}".format(
                stdout
            )
        )

    final_pair = pairs[-1]
    keys_text = final_pair.group(1).strip()
    keys: List[float] = []
    if keys_text:
        for position, token in enumerate(keys_text.split()):
            if re.fullmatch(_FLOAT_TOKEN, token, flags=re.IGNORECASE) is None:
                raise ToolchainError(
                    "Invalid random-key token at position {} in RKO output: {!r}".format(
                        position, token
                    )
                )
            keys.append(float(token))

    return RKOResult(
        keys=keys,
        cost=float(final_pair.group(2)),
        time_to_best=_last_labeled_float(cleaned, "Best time"),
        total_time=_last_labeled_float(cleaned, "Total time"),
        best_metaheuristic=_last_labeled_text(cleaned, "Best MH"),
        instance=_last_labeled_text(cleaned, "Instance"),
        stdout=stdout,
    )


def parse_rko_output(stdout: str) -> Dict[str, object]:
    """Compatibility wrapper returning :func:`parse_rko_stdout` as a dict."""

    return parse_rko_stdout(stdout).as_dict()


def _evaluator_source() -> str:
    return r'''#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "Program/src/Data.h"
#include "Program/src/Problem/Problem.h"

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "usage: evaluate_generated_problem INSTANCE [KEY ...]\n";
        return 2;
    }
    try {
        TProblemData data;
        ReadData(argv[1], data);
        TSol solution;
        solution.rk.reserve(static_cast<std::size_t>(argc - 2));
        if (argc > 2) {
            for (int i = 2; i < argc; ++i) solution.rk.push_back(std::stod(argv[i]));
        } else {
            double key = 0.0;
            while (std::cin >> key) solution.rk.push_back(key);
            if (!std::cin.eof()) {
                std::cerr << "generated evaluator error: invalid key input\n";
                return 4;
            }
        }
        const double cost = Decoder(solution, data);
        std::cout << "RKO_EVALUATOR_RESULT " << std::setprecision(17) << cost << '\n';
        FreeMemoryProblem(data);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "generated evaluator error: " << error.what() << '\n';
        return 3;
    }
}
'''


def _format_key(value: float) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError("Random keys must be real numbers; got {!r}".format(value)) from error
    if not math.isfinite(numeric):
        raise ValueError("Random keys must be finite; got {!r}".format(value))
    return format(numeric, ".17g")


def _last_labeled_float(text: str, label: str) -> Optional[float]:
    matches = re.findall(
        r"(?im)^[ \t]*{}[ \t]*:[ \t]*({})(?=[ \t]*$)".format(
            re.escape(label), _FLOAT_TOKEN
        ),
        text,
    )
    return float(matches[-1]) if matches else None


def _last_labeled_text(text: str, label: str) -> Optional[str]:
    matches = re.findall(
        r"(?im)^[ \t]*{}[ \t]*:[ \t]*(.*?)[ \t]*$".format(re.escape(label)),
        text,
    )
    if not matches:
        return None
    value = matches[-1].strip()
    return value if value else None


def _strip_ansi(text: str) -> str:
    return _ANSI_ESCAPE.sub("", text)


def _paths_overlap(first: Path, second: Path) -> bool:
    try:
        first_text = os.path.normcase(str(first.resolve()))
        second_text = os.path.normcase(str(second.resolve()))
        common = os.path.commonpath((first_text, second_text))
    except ValueError:
        return False
    return common == first_text or common == second_text


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path_text = os.path.normcase(str(path.resolve()))
        parent_text = os.path.normcase(str(parent.resolve()))
        return os.path.commonpath((path_text, parent_text)) == parent_text
    except ValueError:
        return False


def _write_command_logs(
    directory: Path, name: str, completed: subprocess.CompletedProcess
) -> None:
    (directory / (name + ".stdout.log")).write_text(
        completed.stdout or "", encoding="utf-8"
    )
    (directory / (name + ".stderr.log")).write_text(
        completed.stderr or "", encoding="utf-8"
    )


def _run_in_wsl(
    command: Sequence[Union[str, Path]],
    *,
    cwd: Path,
    timeout: Optional[float],
    input_text: Optional[str] = None,
    stream_stdout: bool = False,
) -> subprocess.CompletedProcess:
    command_text = [str(argument) for argument in command]
    host_command: List[str]
    host_cwd: Optional[str]

    if os.name == "nt":
        if shutil.which("wsl.exe") is None:
            raise WSLUnavailableError(
                "wsl.exe was not found. Native RKO builds on Windows require WSL."
            )
        wsl_cwd = _to_wsl_path(cwd)
        wsl_arguments: List[str] = []
        for original, rendered in zip(command, command_text):
            if isinstance(original, Path) or re.match(r"^[A-Za-z]:[\\/]", rendered):
                wsl_arguments.append(_to_wsl_path(Path(rendered)))
            else:
                wsl_arguments.append(rendered)
        shell_command = "cd {} && {}".format(
            shlex.quote(wsl_cwd), shlex.join(wsl_arguments)
        )
        host_command = ["wsl.exe", "bash", "-lc", shell_command]
        host_cwd = None
    else:
        host_command = command_text
        host_cwd = str(cwd)

    try:
        if stream_stdout:
            process = subprocess.Popen(
                host_command,
                cwd=host_cwd,
                text=True,
                encoding="utf-8",
                errors="replace",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.PIPE if input_text else None,
            )
            if input_text and process.stdin:
                process.stdin.write(input_text)
                process.stdin.close()

            stdout_lines = []
            assert process.stdout is not None
            for line in process.stdout:
                if not line.startswith("@@RKO_RESULT_V1@@"):
                    sys.stdout.write(line)
                    sys.stdout.flush()
                stdout_lines.append(line)

            returncode = process.wait(timeout=timeout)
            full_stdout = "".join(stdout_lines)
            completed = subprocess.CompletedProcess(
                args=host_command,
                returncode=returncode,
                stdout=full_stdout,
                stderr="",
            )
        else:
            completed = subprocess.run(
                host_command,
                cwd=host_cwd,
                check=False,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                input=input_text,
                timeout=timeout,
            )
    except FileNotFoundError as error:
        raise WSLUnavailableError(
            "Unable to execute native toolchain command: {}".format(command_text[0])
        ) from error
    except subprocess.TimeoutExpired as error:
        stdout = _coerce_timeout_stream(error.stdout)
        stderr = _coerce_timeout_stream(error.stderr)
        raise ToolchainCommandError(
            "Native toolchain command timed out after {} seconds".format(timeout),
            command=command_text,
            returncode=None,
            stdout=stdout,
            stderr=stderr,
        ) from error

    if completed.returncode != 0:
        hint = "Native toolchain command failed"
        if completed.returncode == 127:
            hint += "; verify that g++ is installed inside WSL"
        raise ToolchainCommandError(
            hint,
            command=command_text,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    return completed


def _to_wsl_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    raw = str(resolved).replace("\\", "/")
    if len(raw) >= 2 and raw[1] == ":":
        drive = raw[0].lower()
        return "/mnt/" + drive + raw[2:]
    try:
        converted = subprocess.run(
            ["wsl.exe", "wslpath", "-a", raw],
            check=False,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=15.0,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        raise WSLUnavailableError("Unable to convert a Windows path through WSL: {}".format(path)) from error
    if converted.returncode != 0 or not converted.stdout.strip():
        raise WSLUnavailableError(
            "WSL could not convert Windows path {!s}: {}".format(
                path, converted.stderr.strip()
            )
        )
    return converted.stdout.strip()


def _coerce_timeout_stream(value: Optional[Union[str, bytes]]) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


__all__ = [
    "EvaluatorResult",
    "PreparedProgram",
    "RKOConfig",
    "RKOResult",
    "ToolchainBuild",
    "ToolchainCommandError",
    "ToolchainError",
    "WSLUnavailableError",
    "build_toolchain",
    "compile_evaluator",
    "compile_rko",
    "default_cpp_program_source",
    "parse_rko_output",
    "parse_rko_stdout",
    "prepare_program",
    "run_evaluator",
    "run_rko",
]
