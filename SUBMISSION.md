# HW6 submission

## AI-use declaration

For Part B, ChatGPT generated the episode runner and result-saving
code, suggested the fixed policy, and helped explain the results.
I ran the program locally in PyCharm. The reported Part B results
come from that local run.
For Part C, ChatGPT generated the value iteration solver, custom
simulator, rule checks and output-saving code. It also helped explain
the results in English. I ran the program locally in PyCharm.
The reported Part C results come from that local run.

## Setup and reproduction

The experiments were run on Windows using PyCharm.

Dependency versions:
- Python: 3.13.2
- Gymnasium: 1.4.0
- NumPy: 2.5.5
- cloudpickle: 3.1.2
- typing-extensions: 4.16.0
- farama-notifications: 0.0.6

Initial installation commands:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install gymnasium
.\.venv\Scripts\python.exe -m pip freeze > requirements.txt
```

To reproduce the environment using the saved versions:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

To run Part B:

```powershell
.\.venv\Scripts\python.exe part_b.py
```

Recorded versions: [part_b_versions.txt](outputs/part_b_versions.txt).

## Part A — Paper calculations

Handwritten calculations:

- [Page 1](evidence/part_a_1.jpeg)
- [Page 2](evidence/part_a_2.jpeg)
- [Page 3](evidence/part_a_3.jpeg)
- [Page 4](evidence/part_a_4.jpeg)

## Part B — FrozenLake

### Fixed policy, written before the experiments

States are numbered from 0 to 15, row by row.

| State | Position (row, column) | Action |
| --- | --- | --- |
| 0 | (1,1) | Down |
| 1 | (1,2) | Right |
| 2 | (1,3) | Down |
| 3 | (1,4) | Left |
| 4 | (2,1) | Down |
| 6 | (2,3) | Down |
| 8 | (3,1) | Right |
| 9 | (3,2) | Down |
| 10 | (3,3) | Down |
| 13 | (4,2) | Right |
| 14 | (4,3) | Right |

States 5, 7, 11 and 12 are holes. State 15 is the goal.
No actions are needed for terminal states.

I chose a route that avoids holes when movement is deterministic:
0 → 4 → 8 → 9 → 13 → 14 → 15.

Actions at other nonterminal states lead toward this route or the goal.
The policy uses the current state, but its action table stays unchanged
across all experiments. Slipping can move the agent away from the route.

### Results

| Slippery? | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| False | random | 11 | 0 | 50 | 0 | 0.0 | 8.14 |
| False | random | 22 | 1 | 49 | 0 | 2.0 | 7.56 |
| False | random | 33 | 0 | 50 | 0 | 0.0 | 6.88 |
| False | fixed | 11 | 50 | 0 | 0 | 100.0 | 6.00 |
| False | fixed | 22 | 50 | 0 | 0 | 100.0 | 6.00 |
| False | fixed | 33 | 50 | 0 | 0 | 100.0 | 6.00 |
| True | random | 11 | 0 | 50 | 0 | 0.0 | 8.06 |
| True | random | 22 | 0 | 50 | 0 | 0.0 | 7.60 |
| True | random | 33 | 0 | 50 | 0 | 0.0 | 7.98 |
| True | fixed | 11 | 1 | 49 | 0 | 2.0 | 4.24 |
| True | fixed | 22 | 2 | 48 | 0 | 4.0 | 5.48 |
| True | fixed | 33 | 1 | 49 | 0 | 2.0 | 4.44 |

### Saved results and traces

- [All 600 episodes](outputs/part_b_episodes.csv)
- [Batch results](outputs/part_b_summary.csv)
- [Fixed policy](outputs/part_b_fixed_policy.txt)
- [Successful fixed-policy episode without slipping](outputs/part_b_trace_slippery_false_fixed_goal.txt)
- [Successful fixed-policy episode with slipping](outputs/part_b_trace_slippery_true_fixed_goal.txt)
- [Failed fixed-policy episode with slipping](outputs/part_b_trace_slippery_true_fixed_hole.txt)
- [Trace index and outcomes not observed](outputs/part_b_trace_index.txt)

### Experiment procedure

I tested two policies with slipping enabled and disabled.
Each condition used three batches of 50 episodes.

The batch seeds were 11, 22 and 33. Episode i used the seed
1000 × batch_seed + i, where i ranged from 0 to 49.
All four conditions used the same episode seed lists.

The random policy used a separate Python random generator,
seeded with the episode seed. Each action had probability 1/4.

The fixed policy used the same action table in every experiment.
The maximum episode length was 100 steps. Terminal outcomes
were checked before time limits, including on the final step.
The final tile was used to distinguish a goal from a hole.

### What slipping changed

Without slipping, the fixed policy reached the goal in all
150 episodes. Every episode took six steps.

With slipping, the fixed policy reached the goal in only
4 of 150 episodes, giving an overall success rate of about 2.67%.
The other 146 episodes ended in holes.

Slipping made the selected direction uncertain. The agent could
leave its planned route and enter a hole. The policy was designed
for a safe deterministic route, so it performed poorly with slipping.

The random policy reached the goal once in 150 episodes without
slipping and zero times with slipping. These small counts do not
provide strong evidence that slipping reduced the random policy's
underlying success probability.

### Variation across batches

The fixed policy without slipping was fully consistent:
all three batches had 100% success and a mean of six steps.

With slipping, its batch success rates were 2%, 4% and 2%.
Its performance was consistently poor, with small differences
caused by random transitions.

The random policy without slipping had success rates of 0%, 2%
and 0%. With slipping, all three batches had 0% success.

No time limits occurred in any of the 600 episodes.
No failures occurred for the fixed policy without slipping.
No successes occurred for the random policy with slipping.

### Interpreting mean steps

Mean steps include both successful and failed episodes.
The slippery fixed policy had shorter average episodes than
the successful deterministic policy because it often fell into
a hole early. Fewer steps alone do not mean better performance.

### Why one successful episode is not enough

One success may happen by chance. For example, the random policy
succeeded once without slipping, but failed in the other 149 episodes.
Repeated experiments show both the success rate and variation
between batches.

### Why these simulations are not training

Neither policy learned from the results. The random policy always
selected actions uniformly. The fixed policy always used the same
action table. No action values or policy parameters were updated.
These experiments evaluated existing policies.


## Part C — Custom solver

### Compare code with paper

The program uses synchronous value iteration. Every update uses
only the values from the previous round.

The calculated V1 map is:

```text
-1    -1    10     0 (G)
-1     #    -1     0 (H)
-1    -1    -1    -1
```

The requested V2 values match my paper calculations:

| Position | Paper V2 | Code V2 |
| --- | ---: | ---: |
| (1,2) | 8 | 8 |
| (1,3) | 10 | 10 |
| (2,3) | 5.21 | 5.21 |

All four action values at each requested position also match.
There were no numerical discrepancies to correct.

[Full paper comparison](outputs/part_c_paper_comparison.txt)

### Values, policies and convergence

The discount factor is 0.9. The stopping condition is a maximum
absolute value change below 0.000001 over a full update.
The iteration safeguard is 10,000 updates.

| Map | Converged? | Iterations | Final maximum change | Start value |
| --- | --- | ---: | ---: | ---: |
| Original | Yes | 12 | 2.16529310482e-07 | 1.535824154 |
| Extended | Yes | 11 | 2.40588123201e-07 | 1.502000000 |

Both maps converged before the safeguard.

The policy is extracted by comparing action values using the final
state values. Actions within 1e-10 of the largest action value are
treated as tied. The tie-breaking order is up, right, down, left.
No tied best actions were found on either map.

Original value map:

```text
4.200000    8.000000   10.000000    0.000000
2.780000       #       5.824176    0.000000
1.535824    2.817582    4.241758    2.817582
```

Original policy:

```text
→  →  →  G
↑  #  ↑  H
→  →  ↑  ←
```

Extended value map:

```text
4.200000    8.000000   10.000000    0.000000
2.780000       #       5.824176    0.000000
1.502000    2.817582    4.241758    2.817582
```

Extended policy:

```text
→  →  →  G
↑  #  ↑  H
↑  →  ↑  ←
```

Saved evidence:

- [Original values and policy](outputs/part_c_original_maps.txt)
- [Extended values and policy](outputs/part_c_extended_maps.txt)
- [Original convergence history](outputs/part_c_original_iterations.csv)
- [Extended convergence history](outputs/part_c_extended_iterations.csv)
- [State-by-state comparison](outputs/part_c_map_comparison.txt)
- [Separate floor rules](outputs/part_c_floor_rules.json)

Floor rewards and movement types are stored in FLOOR_RULES,
separately from value iteration. Sand and water use deterministic
movement. Adding these floors did not require changing the solver.

### Simulation results

Each map was evaluated using three batches of 50 episodes.
The batch seeds were 11, 22 and 33. Episode i used the seed
1000 × batch_seed + i, for i = 0,...,49.

Each episode had a 100-step limit. Discounted return was calculated as:

Return = r0 + 0.9 × r1 + 0.9² × r2 + ...

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |
| original | 11 | 42 | 8 | 0 | 4.98 | 0.891260 |
| original | 22 | 46 | 4 | 0 | 5.04 | 1.914776 |
| original | 33 | 48 | 2 | 0 | 5.04 | 2.492144 |
| extended | 11 | 50 | 0 | 0 | 5.00 | 1.502000 |
| extended | 22 | 50 | 0 | 0 | 5.00 | 1.502000 |
| extended | 33 | 50 | 0 | 0 | 5.00 | 1.502000 |

- [All 300 episodes](outputs/part_c_episodes.csv)
- [Simulation summary](outputs/part_c_summary.csv)


### Explain your results

#### Original map: cost and risk

At the start, the original policy chooses right and moves toward ice.
If no slips occur, its route is:

(3,1) → (3,2) → (3,3) → (2,3) → (1,3) → G.

This route avoids the mud cost but includes a risk of entering the hole.
At ice, the best action is up. It moves up with probability 0.8,
slips into the hole with probability 0.1, and hits the wall with
probability 0.1. After hitting the wall, the agent stays on ice.

The alternative route goes up the left side and crosses mud.
Its discounted return is:

−1 + 0.9 × (−1) + 0.9² × (−3)
+ 0.9³ × (−1) + 0.9⁴ × 10 = 1.502.

The original start value is approximately 1.535824.
This is slightly higher than 1.502, so the solver chooses the risky
route through ice. It maximizes expected discounted reward,
which is not the same as maximizing the probability of success.

#### Extended map: effect of sand

Entering (3,2) now gives −2 instead of −1.

The action value of moving right from the start becomes approximately:

−2 + 0.9 × 2.817582 = 0.535824.

Moving up gives:

−1 + 0.9 × 2.78 = 1.502.

Therefore, the best start action changes from right to up.
The new route is:

(3,1) → (2,1) → (1,1) → (1,2) → (1,3) → G.

This route crosses mud but avoids sand and ice. All movements on
this route are deterministic. Every simulation therefore reaches
the goal in five steps with a discounted return of 1.502.

The start value decreases by approximately 0.033824.
At six decimal places, all other state values remain unchanged,
and no other best action changes.

#### Effect of water and arrival rewards

Water at (3,4) has an arrival reward of −5. It makes entering that
cell less attractive. However, the optimal policy already avoided
entering it on the original map, so it does not change the best actions.

The value of the water cell itself remains approximately 2.817582.
From that cell, the agent moves left onto ordinary floor and receives
−1. The −5 cost applies when a move ends on water, not simply because
the agent starts there.

The same idea explains why the sand cell's own value is unchanged:
its best action moves right onto ordinary floor.

Small differences displayed as −0.000000 come from numerical
approximation and the convergence threshold.

#### Simulation results versus computed values

On the original map, 136 of 150 episodes reached the goal,
giving a success rate of about 90.67%.
Fourteen episodes ended in holes.

The batch mean discounted returns were 0.891260, 1.914776 and
2.492144. Their combined mean was approximately 1.766060.
This differs from the computed start value of 1.535824 because
150 episodes are a finite random sample.

Value iteration estimates the expected discounted return over
the full future under the model. Individual episodes and finite
sample averages do not have to equal that expectation.

The simulations also have a 100-step limit, while the value calculation
has no fixed step limit. No simulated episode reached the time limit,
but this does not prove that longer episodes are impossible.

On the extended map, the selected route is deterministic and ends
after five steps. This explains why all observed returns equal the
computed start value of 1.502.

The extended policy achieved more goals in these runs, although
its optimal expected discounted return is slightly lower. The higher
floor costs changed the preferred balance between cost and risk.


### Rule checks

All implemented rule checks passed.

| Check | Verified behavior |
| --- | --- |
| Boundary | Moving up from (1,1) stays at (1,1), reward −1 |
| Wall from mud | Moving down from (1,2) stays on mud, reward −3 |
| Ice probabilities | 0.8 + 0.1 + 0.1 = 1 for every action |
| Ice wall collision | Choosing left on ice stays there with probability 0.8, reward −1 |
| Mud entry | Reward −3 |
| Sand entry | Reward −2 |
| Water entry | Reward −5 |
| Sand and water movement | Deterministic |
| Goal entry | Reward +10 without an extra floor cost |
| Hole entry | Reward −10 |
| Ice entry | Entering ice from ordinary floor is deterministic |
| Terminal handling | After G or H, further transitions give zero reward |

Positions in the table use one-based coordinates.
Transition tuples in the saved check file use zero-based coordinates.

[Full rule checks](outputs/part_c_rule_checks.txt)
