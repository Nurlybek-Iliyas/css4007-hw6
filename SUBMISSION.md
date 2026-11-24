# HW6 Submission

## AI-use declaration

ChatGPT was used to help understand the assignment, explain the Value Iteration algorithm, debug Python code, and organize the results in this report.

Python and Gymnasium were used to implement and run the experiments. All code was executed and tested locally.

## Setup and reproduction

### Environment

- Python 3.13.2
- Gymnasium 1.4.0
- Operating system: Windows

### Installation

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install gymnasium[toy-text]
```

### Run commands

```bash
python part_b.py
python part_c_base.py
python part_c.py
```

Results are saved in the `outputs/` folder.

---

## Part A — Paper calculations

The paper calculations are included in `part-a-work.pdf`.

### Map

```text
. M . G
. # I H
S . . .
```

The discount factor is gamma = 0.9, and all initial state values are zero.

### V1 values

| State | V1 |
|---|---:|
| (0,0) | -1.00 |
| (0,1) | -1.00 |
| (0,2) | 10.00 |
| (1,0) | -1.00 |
| (1,2) | -1.00 |
| (2,0) | -1.00 |
| (2,1) | -1.00 |
| (2,2) | -1.00 |
| (2,3) | -1.00 |

### V2 calculations

The requested states and their action values are shown below.

**State (0,1)**

| Action | Value |
|---|---:|
| Up | -3.90 |
| Right | 8.00 |
| Down | -3.90 |
| Left | -1.90 |

**Best action: Right.**

**State (0,2)**

| Action | Value |
|---|---:|
| Up | 8.00 |
| Right | 10.00 |
| Down | -1.90 |
| Left | -3.90 |

**Best action: Right.**

**State (1,2) — Ice**

| Action | Value |
|---|---:|
| Up | 5.21 |
| Right | -7.39 |
| Down | -2.71 |
| Left | -0.91 |

**Best action: Up.**

At the ice tile, the intended direction has a probability of 0.8. Each perpendicular direction has a probability of 0.1.

Two iterations are not enough to obtain the final optimal policy because future rewards need more iterations to propagate through the map.

---

## Part B — FrozenLake

### Fixed policy

The fixed policy was written before running the experiments. It does not learn or change during the simulations.

| State | Action |
|---:|---|
| 0 | Down |
| 1 | Left |
| 2 | Left |
| 3 | Left |
| 4 | Down |
| 6 | Down |
| 8 | Right |
| 9 | Down |
| 10 | Left |
| 13 | Right |
| 14 | Right |

### Experimental results

Each condition was tested with 50 episodes in three batches. The random action generator was seeded separately for each episode using the episode seed.

| Slippery | Policy | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Success % | Mean steps |
|---|---|---:|---:|---:|---:|---:|---:|
| False | Random | 11 | 0 | 50 | 0 | 0% | 8.14 |
| False | Random | 22 | 1 | 49 | 0 | 2% | 7.56 |
| False | Random | 33 | 0 | 50 | 0 | 0% | 6.88 |
| False | Fixed | 11 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 22 | 50 | 0 | 0 | 100% | 6.00 |
| False | Fixed | 33 | 50 | 0 | 0 | 100% | 6.00 |
| True | Random | 11 | 0 | 50 | 0 | 0% | 8.06 |
| True | Random | 22 | 0 | 50 | 0 | 0% | 7.60 |
| True | Random | 33 | 0 | 50 | 0 | 0% | 7.98 |
| True | Fixed | 11 | 1 | 49 | 0 | 2% | 5.24 |
| True | Fixed | 22 | 2 | 48 | 0 | 4% | 5.72 |
| True | Fixed | 33 | 1 | 49 | 0 | 2% | 4.96 |

### Explanation

When `is_slippery=False`, the agent moves in the intended direction. The fixed policy reaches the goal in all 150 episodes.

When `is_slippery=True`, the agent can move in a different direction because of slipping. This makes reaching the goal much harder. The fixed policy has only a small number of successful episodes.

Random policy results also change between batches because different random actions are selected.

One successful episode is not enough to prove that a policy is good. More episodes and different seeds are needed to evaluate performance.

These experiments do not train a policy. They only test the performance of random and fixed actions.

### Output files

- `outputs/part_b_results.csv` — episode results.
- `outputs/part_b_summary.csv` — summary for each batch.
- `outputs/successful_trace.txt` — successful episode trace.
- `outputs/failed_trace.txt` — failed episode trace.

---

## Part C — Custom solver

### Comparison with paper calculations

The custom Value Iteration solver was implemented using the same map and rules as Part A.

The first two iterations match the paper calculations.

### V1 values

| State | V1 |
|---|---:|
| (0,0) | -1.00 |
| (0,1) | -1.00 |
| (0,2) | 10.00 |
| (1,0) | -1.00 |
| (1,2) | -1.00 |
| (2,0) | -1.00 |
| (2,1) | -1.00 |
| (2,2) | -1.00 |
| (2,3) | -1.00 |

### Requested V2 values

| State | V2 |
|---|---:|
| (0,1) | 8.00 |
| (0,2) | 10.00 |
| (1,2) | 5.21 |

There were no discrepancies between the paper calculations and the code output.

### Values, policies and convergence

#### Original map

**Final value map:**

```text
[4.2,    8.0,    10.0,    0.0]
[2.78,   #,      5.8242,  0.0]
[1.5358, 2.8176, 4.2418,  2.8176]
```

**Optimal policy:**

```text
[R, R, R, G]
[U, #, U, H]
[R, R, U, L]
```

- Iterations: 12
- Final delta: 0.0000002165
- Stopping threshold: 0.000001

#### Extended map

The extended map includes sand (D) and shallow water (W).

**Final value map:**

```text
[4.2,   8.0,    10.0,   0.0]
[2.78,  #,      5.8242, 0.0]
[1.502, 2.8176, 4.2418, 2.8176]
```

**Optimal policy:**

```text
[R, R, R, G]
[U, #, U, H]
[U, R, U, L]
```

- Iterations: 11
- Final delta: 0.0000002406
- Stopping threshold: 0.000001

Both maps converged because the final delta was smaller than the stopping threshold.

The action order used for tie-breaking is U, R, D, L. If multiple actions have the same maximum value, the first action in this order is selected.

### Output files

- `outputs/part_c_base.txt` — original map values, policy, convergence and simulation.
- `outputs/part_c_extended.txt` — extended map values, policy, convergence and simulation.

### Simulation results

Each map was tested using three batches of 50 episodes.

| Map | Batch seed | Goals /50 | Holes /50 | Time limits /50 | Mean steps | Mean discounted return |
|---|---:|---:|---:|---:|---:|---:|
| Original | 11 | 42 | 8 | 0 | 4.98 | 0.8913 |
| Original | 22 | 46 | 4 | 0 | 5.04 | 1.9148 |
| Original | 33 | 48 | 2 | 0 | 5.04 | 2.4921 |
| Extended | 11 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 22 | 50 | 0 | 0 | 5.00 | 1.5020 |
| Extended | 33 | 50 | 0 | 0 | 5.00 | 1.5020 |

### Explanation of results

The original map has a Right action at the starting state. In the extended map, the starting action changes to Up.

The reason is the sand tile D, which has a reward of -2 instead of -1 for a normal floor tile. This makes the previous route less attractive.

The shallow water tile W has a reward of -5. The optimal policy avoids entering this tile because of its higher cost.

The extended map achieved 150 successful episodes out of 150. The original map had some failures because the optimal policy can still encounter the hole.

The computed values describe expected long-term discounted rewards. Simulations are finite experiments with 50 episodes per batch, so their results can vary. They help evaluate the policy but do not replace the Value Iteration calculations.

### Rule checks

The implementation checks the following rules:

1. **Boundary behavior:** If an action moves outside the map, the agent stays in the same state.
2. **Wall behavior:** The wall is not a valid state. If an action leads to a wall, the agent stays in its current state.
3. **Ice probabilities:** The intended direction has probability 0.8, and the two perpendicular directions have probability 0.1 each. Their total probability is 1.0.
4. **Floor rewards:** Normal floor and start tiles have reward -1, mud has reward -3, ice has reward -1, sand has reward -2, and shallow water has reward -5.
5. **Terminal states:** Goal and hole states have value zero after termination. No additional rewards are received after the episode ends.

The code uses assertions to check ice probabilities, boundary behavior, wall behavior, and terminal rewards.

The program completed successfully and printed:

```text
All model checks passed!
```

---

## Conclusion

In this homework, I implemented and tested a fixed policy, random policy, and custom Value Iteration solver.

The experiments showed that slipping makes FrozenLake much harder. The custom solver successfully converged on both maps. Adding sand changed the optimal action at the starting state, while the shallow water tile was avoided by the optimal policy.
