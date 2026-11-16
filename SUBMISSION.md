# HW6 submission

## AI-use declaration

For Part B, ChatGPT generated the episode runner and result-saving
code, suggested the fixed policy, and helped explain the results.
I ran the program locally in PyCharm. The reported Part B results
come from that local run.

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

Show V1 and the requested V2 values. Explain discrepancies.

### Values, policies and convergence

Link both value/policy maps. Record iteration counts, final value changes and tie-breaking rule.

### Simulation results

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
| --- | --- | --- | --- | --- | --- | --- |

### Explain your results

Which values/actions changed with sand and water? Where did cost and risk matter? How do simulations relate to the computed values?

### Rule checks

Show boundary/wall behavior, slip probabilities, floor rewards and terminal handling.

