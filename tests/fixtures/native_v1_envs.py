"""Small environments that exercise the Native Decoder Language V1.

These classes are fixtures, not special compiler cases.  Each one models a
different decoder/cost shape while keeping the optimization hot path inside the
documented V1 subset.
"""

from __future__ import annotations

import math
import numpy as np


class NativeKnapsackEnv:
    """Threshold decoder with a capacity penalty."""

    def __init__(self) -> None:
        self.tam_solution = 6
        self.capacity = 13.0
        self.profits = [8.0, 5.0, 9.0, 6.0, 7.0, 4.0]
        self.weights = [4.0, 3.0, 6.0, 4.0, 5.0, 2.0]
        self.penalty_factor = 1000.0

    def decoder(self, keys: list[float]) -> list[int]:
        return [1 if key >= 0.5 else 0 for key in keys]

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        total_profit = 0.0
        total_weight = 0.0

        for index, selected in enumerate(solution):
            if selected == 1:
                total_profit += self.profits[index]
                total_weight += self.weights[index]

        penalty = 0.0
        if total_weight > self.capacity:
            penalty = self.penalty_factor * (total_weight - self.capacity)

        return -total_profit + penalty


class NativeTSPEnv:
    """Permutation decoder using the normative stable argsort provider."""

    def __init__(self) -> None:
        self.tam_solution = 5
        self.distance_matrix = [
            [0.0, 2.0, 3.6055512755, 3.1622776602, 2.2360679775],
            [2.0, 0.0, 2.2360679775, 3.1622776602, 3.6055512755],
            [3.6055512755, 2.2360679775, 0.0, 2.2360679775, 4.0],
            [3.1622776602, 3.1622776602, 2.2360679775, 0.0, 2.2360679775],
            [2.2360679775, 3.6055512755, 4.0, 2.2360679775, 0.0],
        ]

    def decoder(self, keys: list[float]) -> list[int]:
        return np.argsort(keys, kind="stable").tolist()

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        total_distance = 0.0
        city_count = len(solution)

        for position in range(city_count):
            origin = solution[position]
            destination = solution[(position + 1) % city_count]
            total_distance += self.distance_matrix[origin][destination]

        return total_distance


class NativeSchedulingEnv:
    """Priority decoder and greedy unrelated-parallel-machine scheduling cost."""

    def __init__(self) -> None:
        self.tam_solution = 6
        self.machine_count = 3
        self.priority_bias = [0.03, -0.04, 0.02, 0.0, -0.01, 0.01]
        self.processing_times = [
            [4.0, 6.0, 5.0],
            [7.0, 3.0, 6.0],
            [5.0, 8.0, 4.0],
            [6.0, 5.0, 7.0],
            [3.0, 6.0, 8.0],
            [8.0, 4.0, 5.0],
        ]

    def _priority(self, job: int, key: float) -> float:
        return key + self.priority_bias[job]

    def _assign_job(
        self,
        job: int,
        machine_ready: list[float],
    ) -> float:
        best_machine = 0
        best_finish = machine_ready[0] + self.processing_times[job][0]
        machine = 1

        while True:
            if machine >= self.machine_count:
                break

            candidate_finish = (
                machine_ready[machine] + self.processing_times[job][machine]
            )
            machine += 1

            if candidate_finish >= best_finish:
                continue

            best_machine = machine - 1
            best_finish = candidate_finish

        machine_ready[best_machine] = best_finish
        return best_finish

    def decoder(self, keys: list[float]) -> list[int]:
        priorities: list[float] = []

        for job in range(len(keys)):
            priorities.append(self._priority(job, keys[job]))

        return np.argsort(priorities, kind="stable").tolist()

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        machine_ready = [0.0] * self.machine_count

        for position in range(len(solution)):
            job = solution[position]
            assigned_finish = self._assign_job(job, machine_ready)

        final_ready = machine_ready.copy()
        makespan = final_ready[0]
        for machine in range(1, self.machine_count):
            if final_ready[machine] > makespan:
                makespan = final_ready[machine]

        return makespan


class NativeNumericSemanticsEnv:
    """Numeric/runtime fixture for round, math, pop and negative indexing."""

    def __init__(self) -> None:
        self.tam_solution = 4
        self.round_half = 2.5
        self.round_decimal = 2.675
        self.round_boundary = 0.000005

    def _pop_and_restore(self, values: list[float]) -> float:
        value = values.pop()
        values.append(value)
        return value

    def decoder(self, keys: list[float]) -> list[float]:
        return keys.copy()

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        values: list[float] = solution.copy()
        restored = self._pop_and_restore(values)
        total = 0.0
        index = len(values) - 1
        while index >= 0:
            total += values[index]
            index -= 1

        whole = round(self.round_half)
        decimal = round(self.round_decimal, 2)
        boundary = round(self.round_boundary, 5)
        floor_value = math.floor(-1.2)
        ceil_value = math.ceil(1.2)
        math_value = (
            math.sqrt(4.0)
            + math.log(1.0)
            + math.exp(0.0)
            + math.sin(0.25)
            + math.cos(0.25)
            + math.tan(0.25)
        )
        modulo_value = -5.5 % 2
        floor_division = -5.5 // 2
        floor_precision_edge = 1.0 // 0.1
        int_modulo_edge = (-9223372036854775808) % 3
        return (
            total
            + restored * 0.0
            + float(whole + floor_value + ceil_value)
            + decimal
            + boundary
            + math_value
            + modulo_value
            + floor_division
            + floor_precision_edge
            + float(int_modulo_edge)
        )


def make_native_knapsack_env() -> NativeKnapsackEnv:
    return NativeKnapsackEnv()


def make_native_tsp_env() -> NativeTSPEnv:
    return NativeTSPEnv()


def make_native_scheduling_env() -> NativeSchedulingEnv:
    return NativeSchedulingEnv()


def make_native_numeric_semantics_env() -> NativeNumericSemanticsEnv:
    return NativeNumericSemanticsEnv()


NATIVE_V1_ENV_FACTORIES = (
    make_native_knapsack_env,
    make_native_tsp_env,
    make_native_scheduling_env,
    make_native_numeric_semantics_env,
)


class InvalidDefaultArgsortEnv:
    """Negative fixture: V1 never guesses NumPy's default sorting semantics."""

    def __init__(self) -> None:
        self.tam_solution = 3

    def decoder(self, keys: list[float]) -> list[int]:
        return np.argsort(keys).tolist()

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(solution[0])


class InvalidSharedMutationEnv:
    """Negative fixture: evaluation may not mutate the frozen environment."""

    def __init__(self) -> None:
        self.tam_solution = 2
        self.counter = 0

    def decoder(self, keys: list[float]) -> list[int]:
        self.counter += 1
        return [0, 1]

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(solution[0])


class InvalidBorrowedAliasEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[float]:
        borrowed: list[float] = keys
        return borrowed

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        return solution[0]


class InvalidFrozenHelperMutationEnv:
    def __init__(self) -> None:
        self.tam_solution = 2
        self.bias = [0.0, 0.0]

    def _bump(self, values: list[float]) -> None:
        values[0] += 1.0

    def decoder(self, keys: list[float]) -> list[float]:
        self._bump(self.bias)
        return keys.copy()

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        return solution[0]


class InvalidRootHelperMutationEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def _bump(self, values: list[float]) -> None:
        values[0] += 1.0

    def decoder(self, keys: list[float]) -> list[float]:
        self._bump(keys)
        return keys.copy()

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        return solution[0]


class InvalidConditionalInitializationEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[int]:
        if keys[0] > 0.5:
            choice = 1
        return [choice, 0]

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(solution[0])


class InvalidUninitializedAnnotationEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[int]:
        choice: int
        return [choice, 0]

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(solution[0])


class InvalidRangeStepEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[int]:
        result: list[int] = []
        for index in range(0, len(keys), 0):
            result.append(index)
        return result

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(len(solution))


class InvalidMissingReturnEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[int]:
        if keys[0] > 0.5:
            return [1, 0]

    def cost(self, solution: list[int], final_solution: bool = False) -> float:
        return float(solution[0])


class InvalidTupleAliasEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def _bump(self, values: list[float]) -> None:
        values[0] += 1.0

    def decoder(self, keys: list[float]) -> list[float]:
        box = (keys,)
        self._bump(box[0])
        return keys.copy()

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        return solution[0]


class InvalidNestedEffectEnv:
    def __init__(self) -> None:
        self.tam_solution = 2

    def decoder(self, keys: list[float]) -> list[float]:
        values: list[float] = keys.copy()
        combined = values.pop() + values.pop()
        return [combined]

    def cost(self, solution: list[float], final_solution: bool = False) -> float:
        return solution[0]
