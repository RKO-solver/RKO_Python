from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from rko.compiler import (
    LANGUAGE_VERSION,
    RKOConfig,
    ToolchainError,
    analyze_environment,
    compile_environment,
)
from rko.compiler.build_make import parse_rko_stdout, prepare_program

from tests.fixtures.native_v1_envs import (
    InvalidBorrowedAliasEnv,
    InvalidConditionalInitializationEnv,
    InvalidDefaultArgsortEnv,
    InvalidFrozenHelperMutationEnv,
    InvalidMissingReturnEnv,
    InvalidNestedEffectEnv,
    InvalidRangeStepEnv,
    InvalidRootHelperMutationEnv,
    InvalidSharedMutationEnv,
    InvalidTupleAliasEnv,
    InvalidUninitializedAnnotationEnv,
    NATIVE_V1_ENV_FACTORIES,
)


class NativeCompilerV1Tests(unittest.TestCase):
    def test_all_positive_fixtures_analyze_and_generate(self) -> None:
        for factory in NATIVE_V1_ENV_FACTORIES:
            with self.subTest(environment=factory.__name__):
                environment = factory()
                report = analyze_environment(environment)
                self.assertTrue(report["supported"], report)
                self.assertEqual(report["language_version"], LANGUAGE_VERSION)
                self.assertIn("decoder", report["function_order"])
                self.assertIn("cost", report["function_order"])

                with tempfile.TemporaryDirectory(prefix="rko_native_v1_") as temporary:
                    artifact = compile_environment(environment, temporary)
                    header = artifact.problem_header.read_text(encoding="utf-8")
                    self.assertIn("RKO_NATIVE_V1_PROBLEM_H", header)
                    self.assertIn("double Decoder(TSol& s", header)
                    self.assertTrue(artifact.instance_data.is_file())
                    self.assertTrue(artifact.schema.is_file())
                    self.assertTrue(artifact.typed_ir.is_file())
                    manifest = json.loads(
                        (Path(temporary) / "manifest.json").read_text(encoding="utf-8")
                    )
                    self.assertEqual(manifest["language_version"], LANGUAGE_VERSION)

    def test_default_numpy_argsort_is_rejected_with_adaptation(self) -> None:
        report = analyze_environment(InvalidDefaultArgsortEnv())
        self.assertFalse(report["supported"])
        diagnostic = report["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "RKO-NATIVE-NUMPY-003")
        self.assertIn("kind='stable'", diagnostic["adaptation"])

    def test_shared_environment_mutation_is_rejected(self) -> None:
        report = analyze_environment(InvalidSharedMutationEnv())
        self.assertFalse(report["supported"])
        diagnostic = report["diagnostics"][0]
        self.assertEqual(diagnostic["code"], "RKO-NATIVE-EFFECT-001")
        self.assertIn("self.counter", diagnostic["message"])

    def test_python_fixture_reference_values(self) -> None:
        keys = [0.99, 0.01, 0.7, 0.3, 0.6, 0.2]
        expected = [1976.0, 15.0038969132, 9.0, 18.641668302186204]
        for factory, expected_cost in zip(NATIVE_V1_ENV_FACTORIES, expected):
            environment = factory()
            selected_keys = keys[: environment.tam_solution]
            actual = float(environment.cost(environment.decoder(selected_keys)))
            self.assertAlmostEqual(actual, expected_cost, places=10)

    def test_effect_and_control_flow_diagnostics(self) -> None:
        cases = (
            (InvalidBorrowedAliasEnv, "RKO-NATIVE-EFFECT-003"),
            (InvalidFrozenHelperMutationEnv, "RKO-NATIVE-EFFECT-011"),
            (InvalidRootHelperMutationEnv, "RKO-NATIVE-EFFECT-010"),
            (InvalidConditionalInitializationEnv, "RKO-NATIVE-NAME-001"),
            (InvalidUninitializedAnnotationEnv, "RKO-NATIVE-NAME-001"),
            (InvalidRangeStepEnv, "RKO-NATIVE-CALL-019"),
            (InvalidMissingReturnEnv, "RKO-NATIVE-CONTROL-001"),
            (InvalidTupleAliasEnv, "RKO-NATIVE-EFFECT-015"),
            (InvalidNestedEffectEnv, "RKO-NATIVE-EFFECT-017"),
        )
        for environment_type, expected_code in cases:
            with self.subTest(environment=environment_type.__name__):
                report = analyze_environment(environment_type())
                self.assertFalse(report["supported"], report)
                self.assertEqual(report["diagnostics"][0]["code"], expected_code)

    def test_machine_result_protocol_preserves_exact_keys(self) -> None:
        stdout = """
Best MH: MultiStart
Instance: instance.rko-data
sol: 0.123 0.988
ofv: 10.12346
Total time: 1.000
Best time: 0.500
@@RKO_RESULT_V1@@ 2 10.123456789012345 0.12345678901234566 0.98765432109876539
"""
        result = parse_rko_stdout(stdout)
        self.assertFalse(result.output_is_lossy)
        self.assertEqual(result.protocol_version, "RKO_RESULT_V1")
        self.assertEqual(result.cost, 10.123456789012345)
        self.assertEqual(result.keys, [0.12345678901234566, 0.9876543210987654])

    def test_malformed_machine_result_never_falls_back_to_lossy_text(self) -> None:
        stdout = "sol: 0.123\nofv: 1.00000\n@@RKO_RESULT_V1@@ 2 1.0 0.123\n"
        with self.assertRaises(ToolchainError):
            parse_rko_stdout(stdout)

    def test_legacy_result_is_explicitly_lossy(self) -> None:
        result = parse_rko_stdout(
            "Instance: /tmp/@@RKO_RESULT_V1@@/instance.rko-data\n"
            "sol: 0.123 0.988\nofv: 10.12346\n"
        )
        self.assertTrue(result.output_is_lossy)
        self.assertIsNone(result.protocol_version)

    def test_rko_config_rejects_duplicate_algorithms(self) -> None:
        with self.assertRaises(ValueError):
            RKOConfig(algorithms=("MultiStart", "MultiStart")).render()
        with self.assertRaises(TypeError):
            RKOConfig(max_runs=1.5).render()  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            RKOConfig(restart=1.5).render()

    def test_toolchain_never_deletes_program_with_only_a_stale_outer_marker(self) -> None:
        with tempfile.TemporaryDirectory(prefix="rko_native_ownership_") as temporary:
            root = Path(temporary)
            generated = root / "generated"
            generated.mkdir()
            (generated / "Problem.h").write_text("// generated\n", encoding="utf-8")
            (generated / "instance.rko-data").write_text("RKO_NATIVE_DATA_V1\n", encoding="utf-8")
            source = root / "source-program"
            source.mkdir()
            build = root / "build"
            existing_program = build / "Program"
            existing_program.mkdir(parents=True)
            sentinel = existing_program / "user-data.txt"
            sentinel.write_text("preserve me\n", encoding="utf-8")
            (build / ".rko-native-build-v1").write_text(
                ".rko-native-build-v1\n", encoding="utf-8"
            )

            with self.assertRaises(ToolchainError):
                prepare_program(generated, build, cpp_program_source=source)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve me\n")


if __name__ == "__main__":
    unittest.main()
