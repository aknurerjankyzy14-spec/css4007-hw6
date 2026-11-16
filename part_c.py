import csv
import json
import random
from pathlib import Path
from statistics import mean


OUTPUT = Path(__file__).resolve().parent / "outputs"

GAMMA = 0.9
TOLERANCE = 0.000001
MAX_ITERATIONS = 10000
MAX_STEPS = 100
BATCH_SEEDS = (11, 22, 33)

MAPS = {
    "original": (
        ".M.G",
        ".#IH",
        "S...",
    ),
    "extended": (
        ".M.G",
        ".#IH",
        "SD.W",
    ),
}

# Floor rules are separate from the solver.
# Each rule defines the arrival reward and movement type.
FLOOR_RULES = {
    ".": {"reward": -1, "movement": "normal"},
    "S": {"reward": -1, "movement": "normal"},
    "M": {"reward": -3, "movement": "normal"},
    "I": {"reward": -1, "movement": "ice"},
    "D": {"reward": -2, "movement": "normal"},
    "W": {"reward": -5, "movement": "normal"},
    "G": {"reward": 10, "movement": "terminal"},
    "H": {"reward": -10, "movement": "terminal"},
}

# Order also defines tie-breaking: up, right, down, left.
ACTIONS = ("up", "right", "down", "left")
DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))
ARROWS = ("↑", "→", "↓", "←")
TIE_TOLERANCE = 1e-10


class GridWorld:
    def __init__(self, grid):
        self.grid = grid
        self.height = len(grid)
        self.width = len(grid[0])

        self.states = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if grid[r][c] != "#"
        ]

        self.start = next(
            s for s in self.states if self.tile(s) == "S"
        )

        self.nonterminal = [
            s for s in self.states if not self.is_terminal(s)
        ]

    def tile(self, state):
        r, c = state
        return self.grid[r][c]

    def is_terminal(self, state):
        return FLOOR_RULES[self.tile(state)]["movement"] == "terminal"

    def destination(self, state, direction):
        r, c = state
        dr, dc = DELTAS[direction]
        nr, nc = r + dr, c + dc

        if not (0 <= nr < self.height and 0 <= nc < self.width):
            return state

        if self.grid[nr][nc] == "#":
            return state

        return nr, nc

    def transitions(self, state, action):
        # Absorbing terminal state with zero further rewards.
        if self.is_terminal(state):
            return [(1.0, state, 0.0)]

        movement = FLOOR_RULES[self.tile(state)]["movement"]

        if movement == "ice":
            directions = (
                (0.8, action),
                (0.1, (action - 1) % 4),
                (0.1, (action + 1) % 4),
            )
        else:
            directions = ((1.0, action),)

        results = []

        for probability, direction in directions:
            next_state = self.destination(state, direction)
            reward = FLOOR_RULES[self.tile(next_state)]["reward"]
            results.append((probability, next_state, reward))

        return results

    def step(self, state, action, rng):
        transitions = self.transitions(state, action)
        draw = rng.random()
        cumulative = 0.0

        for probability, next_state, reward in transitions:
            cumulative += probability
            if draw < cumulative:
                return next_state, reward, self.is_terminal(next_state)

        # Floating-point fallback.
        _, next_state, reward = transitions[-1]
        return next_state, reward, self.is_terminal(next_state)


def action_value(world, state, action, values):
    return sum(
        probability * (reward + GAMMA * values[next_state])
        for probability, next_state, reward
        in world.transitions(state, action)
    )


def update_values(world, old_values):
    # New dictionary ensures synchronous updates:
    # every calculation uses only the previous round.
    new_values = {s: 0.0 for s in world.states}

    for state in world.nonterminal:
        new_values[state] = max(
            action_value(world, state, action, old_values)
            for action in range(4)
        )

    return new_values


def solve(world):
    values = {s: 0.0 for s in world.states}
    history = []

    for iteration in range(1, MAX_ITERATIONS + 1):
        new_values = update_values(world, values)

        change = max(
            abs(new_values[s] - values[s])
            for s in world.states
        )

        history.append({
            "iteration": iteration,
            "max_change": change,
        })

        values = new_values

        if change < TOLERANCE:
            break

    converged = change < TOLERANCE
    policy = {}
    ties = {}

    for state in world.nonterminal:
        scores = [
            action_value(world, state, action, values)
            for action in range(4)
        ]
        best = max(scores)

        tied = [
            action
            for action, score in enumerate(scores)
            if abs(score - best) <= TIE_TOLERANCE
        ]

        policy[state] = tied[0]
        ties[state] = tied

    return values, policy, ties, history, converged


def value_map(world, values):
    lines = []

    for r in range(world.height):
        cells = []
        for c in range(world.width):
            if world.grid[r][c] == "#":
                cells.append(f"{'#':>11}")
            else:
                cells.append(f"{values[(r, c)]:11.6f}")
        lines.append(" ".join(cells))

    return "\n".join(lines)


def policy_map(world, policy):
    lines = []

    for r in range(world.height):
        cells = []
        for c in range(world.width):
            tile = world.grid[r][c]
            if tile in "#GH":
                cells.append(tile)
            else:
                cells.append(ARROWS[policy[(r, c)]])
        lines.append("  ".join(cells))

    return "\n".join(lines)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_checks():
    original = GridWorld(MAPS["original"])
    extended = GridWorld(MAPS["extended"])
    messages = []

    def check(label, condition, evidence):
        if not condition:
            raise AssertionError(f"{label}: {evidence}")
        messages.append(f"PASS: {label}\n  {evidence}")

    # Internal coordinates start at zero.
    result = original.transitions((0, 0), 0)
    check(
        "Boundary behavior",
        result == [(1.0, (0, 0), -1)],
        f"Up from (1,1): {result}",
    )

    result = original.transitions((0, 1), 2)
    check(
        "Wall collision on mud",
        result == [(1.0, (0, 1), -3)],
        f"Down from mud (1,2): {result}",
    )

    for action in range(4):
        result = original.transitions((1, 2), action)
        probabilities = [item[0] for item in result]

        check(
            f"Ice probabilities: {ACTIONS[action]}",
            probabilities == [0.8, 0.1, 0.1]
            and abs(sum(probabilities) - 1.0) < 1e-12,
            str(result),
        )

    result = original.transitions((1, 2), 3)
    check(
        "Ice collision with wall",
        result[0] == (0.8, (1, 2), -1),
        f"Chosen left on ice: {result}",
    )

    reward_cases = [
        ("Mud reward", original, (0, 0), 1, (0, 1), -3),
        ("Sand reward", extended, (2, 0), 1, (2, 1), -2),
        ("Water reward", extended, (2, 2), 1, (2, 3), -5),
        ("Goal reward", original, (0, 2), 1, (0, 3), 10),
        ("Hole reward", original, (2, 3), 0, (1, 3), -10),
        ("Entering ice", original, (0, 2), 2, (1, 2), -1),
    ]

    for label, world, state, action, destination, reward in reward_cases:
        result = world.transitions(state, action)
        check(
            label,
            result == [(1.0, destination, reward)],
            str(result),
        )

    for state in ((2, 1), (2, 3)):
        result = extended.transitions(state, 3)
        check(
            f"Deterministic movement from {extended.tile(state)}",
            len(result) == 1 and result[0][0] == 1.0,
            str(result),
        )

    for state in ((0, 3), (1, 3)):
        for action in range(4):
            result = original.transitions(state, action)
            check(
                f"No rewards after {original.tile(state)}: "
                f"{ACTIONS[action]}",
                result == [(1.0, state, 0.0)],
                str(result),
            )

    messages.insert(
        0,
        "Transition tuples below use zero-based coordinates."
    )
    return "\n".join(messages)


def compare_with_paper():
    world = GridWorld(MAPS["original"])
    zero = {s: 0.0 for s in world.states}

    v1 = update_values(world, zero)
    v2 = update_values(world, v1)

    expected_v1 = {
        state: (
            0.0 if world.is_terminal(state)
            else 10.0 if state == (0, 2)
            else -1.0
        )
        for state in world.states
    }

    expected_v2 = {
        (0, 1): 8.0,
        (0, 2): 10.0,
        (1, 2): 5.21,
    }

    for state, expected in expected_v1.items():
        assert abs(v1[state] - expected) < 1e-10

    lines = ["V1:", value_map(world, v1), "", "Requested V2:"]

    for state, expected in expected_v2.items():
        actual = v2[state]
        assert abs(actual - expected) < 1e-10

        r, c = state
        lines.append(
            f"({r + 1},{c + 1}): "
            f"code={actual:.6f}, paper={expected:.6f}"
        )

        for action in range(4):
            score = action_value(world, state, action, v1)
            lines.append(f"  {ACTIONS[action]}: {score:.6f}")

    lines.append("\nAll values match the Part A calculations.")
    return "\n".join(lines)


def simulate(world, policy, map_name):
    rows = []
    summaries = []

    for batch_seed in BATCH_SEEDS:
        batch = []

        for i in range(50):
            episode_seed = 1000 * batch_seed + i
            rng = random.Random(episode_seed)
            state = world.start

            total_reward = 0.0
            discounted_return = 0.0
            outcome = "time limit"

            for step in range(1, MAX_STEPS + 1):
                action = policy[state]
                state, reward, terminated = world.step(
                    state, action, rng
                )

                total_reward += reward
                discounted_return += GAMMA ** (step - 1) * reward

                if terminated:
                    outcome = (
                        "goal" if world.tile(state) == "G" else "hole"
                    )
                    break

            row = {
                "map": map_name,
                "batch_seed": batch_seed,
                "episode_seed": episode_seed,
                "steps": step,
                "total_reward": total_reward,
                "discounted_return": discounted_return,
                "outcome": outcome,
            }

            batch.append(row)
            rows.append(row)

        goals = sum(r["outcome"] == "goal" for r in batch)
        holes = sum(r["outcome"] == "hole" for r in batch)
        limits = sum(r["outcome"] == "time limit" for r in batch)

        assert goals + holes + limits == 50

        summaries.append({
            "map": map_name,
            "batch_seed": batch_seed,
            "goals": goals,
            "holes": holes,
            "time_limits": limits,
            "mean_steps": mean(r["steps"] for r in batch),
            "mean_discounted_return": mean(
                r["discounted_return"] for r in batch
            ),
        })

    return rows, summaries


def main():
    OUTPUT.mkdir(exist_ok=True)

    checks = run_checks()
    (OUTPUT / "part_c_rule_checks.txt").write_text(
        checks + "\n", encoding="utf-8"
    )
    print("All rule checks passed.")

    paper = compare_with_paper()
    (OUTPUT / "part_c_paper_comparison.txt").write_text(
        paper + "\n", encoding="utf-8"
    )
    print("\n" + paper)

    (OUTPUT / "part_c_floor_rules.json").write_text(
        json.dumps(FLOOR_RULES, indent=2) + "\n",
        encoding="utf-8",
    )

    all_episodes = []
    all_summaries = []
    solved = {}

    for map_name, grid in MAPS.items():
        world = GridWorld(grid)
        values, policy, ties, history, converged = solve(world)
        solved[map_name] = (world, values, policy)

        report = [
            f"Map: {map_name}",
            "\n".join(grid),
            f"\nConverged: {converged}",
            f"Iterations: {len(history)}",
            f"Final maximum change: {history[-1]['max_change']:.12g}",
            f"Start value: {values[world.start]:.9f}",
            "\nValues:",
            value_map(world, values),
            "\nPolicy:",
            policy_map(world, policy),
            "\nTie rule: first of up, right, down, left.",
            f"Tie tolerance: {TIE_TOLERANCE}",
            "\nTied best actions:",
        ]

        tied_states = [
            state for state in world.nonterminal
            if len(ties[state]) > 1
        ]

        if not tied_states:
            report.append("None.")

        for state in tied_states:
            r, c = state
            names = ", ".join(ACTIONS[a] for a in ties[state])
            report.append(f"({r + 1},{c + 1}): {names}")

        if not converged:
            report.append(
                "WARNING: safeguard reached before convergence."
            )

        report_text = "\n".join(report)

        (OUTPUT / f"part_c_{map_name}_maps.txt").write_text(
            report_text + "\n", encoding="utf-8"
        )
        save_csv(
            OUTPUT / f"part_c_{map_name}_iterations.csv",
            history,
        )

        print("\n" + report_text)

        episodes, summaries = simulate(world, policy, map_name)
        all_episodes.extend(episodes)
        all_summaries.extend(summaries)

    save_csv(OUTPUT / "part_c_episodes.csv", all_episodes)
    save_csv(OUTPUT / "part_c_summary.csv", all_summaries)

    table = [
        "| Map | Batch seed | Goals /50 | Holes /50 | "
        "Time limits /50 | Mean steps | Mean discounted return |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in all_summaries:
        table.append(
            f"| {row['map']} | {row['batch_seed']} | "
            f"{row['goals']} | {row['holes']} | "
            f"{row['time_limits']} | {row['mean_steps']:.2f} | "
            f"{row['mean_discounted_return']:.6f} |"
        )

    table_text = "\n".join(table)
    (OUTPUT / "part_c_summary.md").write_text(
        table_text + "\n", encoding="utf-8"
    )
    print("\n" + table_text)

    original, old_values, old_policy = solved["original"]
    _, new_values, new_policy = solved["extended"]

    differences = [
        "Position | Original V | Extended V | Change | "
        "Original action | Extended action"
    ]

    for state in original.nonterminal:
        r, c = state
        differences.append(
            f"({r + 1},{c + 1}) | "
            f"{old_values[state]:.6f} | "
            f"{new_values[state]:.6f} | "
            f"{new_values[state] - old_values[state]:+.6f} | "
            f"{ACTIONS[old_policy[state]]} | "
            f"{ACTIONS[new_policy[state]]}"
        )

    comparison = "\n".join(differences)
    (OUTPUT / "part_c_map_comparison.txt").write_text(
        comparison + "\n", encoding="utf-8"
    )
    print("\n" + comparison)

    print(f"\nSaved {len(all_episodes)} episodes.")
    print(f"Output folder: {OUTPUT}")


if __name__ == "__main__":
    main()