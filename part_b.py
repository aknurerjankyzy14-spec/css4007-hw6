import csv
import platform
import random
import re
from importlib.metadata import version
from pathlib import Path
from statistics import mean

import gymnasium as gym


OUTPUT = Path(__file__).resolve().parent / "outputs"
BATCH_SEEDS = (11, 22, 33)
EPISODES = 50
MAX_STEPS = 100

# FrozenLake actions: 0=left, 1=down, 2=right, 3=up
ACTION_NAMES = {
    0: "left",
    1: "down",
    2: "right",
    3: "up",
}

# Designed before running experiments. Never updated during runs.
FIXED_POLICY = {
    0: 1,
    1: 2,
    2: 1,
    3: 0,
    4: 1,
    6: 1,
    8: 2,
    9: 1,
    10: 1,
    13: 2,
    14: 2,
}


def render_text(env):
    """Remove terminal colour codes from the text rendering."""
    return re.sub(r"\x1b\[[0-9;]*m", "", env.render())


def run_episode(env, policy, batch_seed, episode_seed, condition):
    state, _ = env.reset(seed=episode_seed)

    # Separate seeded generator for uniformly random actions.
    rng = random.Random(episode_seed)

    total_reward = 0.0
    outcome = "time limit"

    trace = [
        f"Condition: {condition}",
        f"Policy: {policy}",
        f"Batch seed: {batch_seed}",
        f"Episode seed: {episode_seed}",
        f"Initial state: {state}",
        render_text(env),
    ]

    for steps in range(1, MAX_STEPS + 1):
        if policy == "random":
            action = rng.randrange(4)
        else:
            action = FIXED_POLICY[int(state)]

        old_state = int(state)
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)

        trace.append(
            f"Step {steps}: state={old_state}, "
            f"action={ACTION_NAMES[action]}, next_state={state}, "
            f"reward={reward}, terminated={terminated}, "
            f"truncated={truncated}"
        )
        trace.append(render_text(env))

        # Check termination FIRST, including on step 100.
        if terminated:
            tile = env.unwrapped.desc.reshape(-1)[int(state)]
            if tile == b"G":
                outcome = "goal"
            elif tile == b"H":
                outcome = "hole"
            else:
                raise RuntimeError("Unexpected terminal tile.")
            break

        if truncated:
            outcome = "time limit"
            break

    trace.append(f"Outcome: {outcome}")
    trace.append(f"Steps: {steps}")
    trace.append(f"Total reward: {total_reward}")

    row = {
        "condition": condition,
        "batch_seed": batch_seed,
        "episode_seed": episode_seed,
        "policy": policy,
        "steps": steps,
        "total_reward": total_reward,
        "outcome": outcome,
    }

    return row, "\n".join(trace)


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(exist_ok=True)

    versions = [f"Python: {platform.python_version()}"]
    for package in (
        "gymnasium",
        "numpy",
        "cloudpickle",
        "typing-extensions",
        "farama-notifications",
    ):
        versions.append(f"{package}: {version(package)}")

    (OUTPUT / "part_b_versions.txt").write_text(
        "\n".join(versions) + "\n",
        encoding="utf-8",
    )

    # Save the policy before any episodes are run.
    policy_text = "\n".join(
        f"State {state}: {ACTION_NAMES[action]}"
        for state, action in sorted(FIXED_POLICY.items())
    )
    (OUTPUT / "part_b_fixed_policy.txt").write_text(
        policy_text + "\n",
        encoding="utf-8",
    )

    all_rows = []
    summaries = []
    trace_index = []

    for slippery in (False, True):
        for policy in ("random", "fixed"):
            condition = f"slippery_{str(slippery).lower()}_{policy}"
            saved_outcomes = set()

            env = gym.make(
                "FrozenLake-v1",
                map_name="4x4",
                is_slippery=slippery,
                render_mode="ansi",
                max_episode_steps=MAX_STEPS,
            )

            try:
                for batch_seed in BATCH_SEEDS:
                    batch_rows = []

                    for i in range(EPISODES):
                        episode_seed = 1000 * batch_seed + i

                        row, trace = run_episode(
                            env,
                            policy,
                            batch_seed,
                            episode_seed,
                            condition,
                        )
                        batch_rows.append(row)
                        all_rows.append(row)

                        # Save the first trace of each observed outcome
                        # for every condition.
                        outcome = row["outcome"]
                        if outcome not in saved_outcomes:
                            filename = (
                                f"part_b_trace_{condition}_"
                                f"{outcome.replace(' ', '_')}.txt"
                            )
                            (OUTPUT / filename).write_text(
                                trace,
                                encoding="utf-8",
                            )
                            saved_outcomes.add(outcome)
                            trace_index.append(f"{condition}: {filename}")

                    goals = sum(r["outcome"] == "goal" for r in batch_rows)
                    holes = sum(r["outcome"] == "hole" for r in batch_rows)
                    limits = sum(
                        r["outcome"] == "time limit" for r in batch_rows
                    )

                    assert goals + holes + limits == EPISODES

                    summaries.append({
                        "slippery": slippery,
                        "policy": policy,
                        "batch_seed": batch_seed,
                        "goals": goals,
                        "holes": holes,
                        "time_limits": limits,
                        "success_percent": 100 * goals / EPISODES,
                        "mean_steps": mean(r["steps"] for r in batch_rows),
                    })
            finally:
                env.close()

            for outcome in ("goal", "hole", "time limit"):
                if outcome not in saved_outcomes:
                    trace_index.append(
                        f"{condition}: {outcome} was not observed "
                        f"in {len(BATCH_SEEDS) * EPISODES} episodes."
                    )

    assert len(all_rows) == 600
    assert len(summaries) == 12

    save_csv(OUTPUT / "part_b_episodes.csv", all_rows)
    save_csv(OUTPUT / "part_b_summary.csv", summaries)

    table = [
        "| Slippery? | Policy | Batch seed | Goals /50 | "
        "Holes /50 | Time limits /50 | Success % | Mean steps |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for row in summaries:
        table.append(
            f"| {row['slippery']} | {row['policy']} | "
            f"{row['batch_seed']} | {row['goals']} | "
            f"{row['holes']} | {row['time_limits']} | "
            f"{row['success_percent']:.1f} | "
            f"{row['mean_steps']:.2f} |"
        )

    table_text = "\n".join(table)
    (OUTPUT / "part_b_summary.md").write_text(
        table_text + "\n",
        encoding="utf-8",
    )
    (OUTPUT / "part_b_trace_index.txt").write_text(
        "\n".join(trace_index) + "\n",
        encoding="utf-8",
    )

    print("\n".join(versions))
    print()
    print(table_text)
    print(f"\nSaved {len(all_rows)} episodes.")
    print(f"Output folder: {OUTPUT}")


if __name__ == "__main__":
    main()