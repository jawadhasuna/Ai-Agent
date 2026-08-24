# Maze Solver

An agent that learns to escape a maze by trial and error. Nobody tells it where the
exit is or which way to walk — it finds out by walking into walls a few thousand
times and remembering what hurt.

Tabular Q-learning, written from scratch in about 60 lines. No libraries beyond
NumPy and Matplotlib.

---

## The maze

A 6×6 grid. `0` is open floor, `1` is a wall. Start is the top-left corner, the goal
is bottom-right.

```
S . . . . .          0 0 0 0 0 0
█ █ . █ █ .          1 1 0 1 1 0
. . . . █ .          0 0 0 0 1 0
. █ █ . █ .          0 1 1 0 1 0
. █ . . . .          0 1 0 0 0 0
. . . █ █ G          0 0 0 1 1 0
```

The shortest possible route is **10 moves** — along the top row, then straight down
the right-hand column.

## How it learns

The agent knows three things: where it is, which four directions exist, and whether
the last move felt good. That is all.

**Reward.** Reaching the goal is worth `+100`. Every other move costs `-1`. That
small penalty is what makes it prefer short routes — dawdling is expensive, so the
agent that wanders for 80 steps ends up worse off than the one that goes straight
there.

**The table.** It keeps a score for every combination of square and direction —
36 squares × 4 directions = **144 numbers**, all starting at zero. That table is the
entire brain.

**The update.** After each move it nudges one of those 144 numbers:

```python
Q[(state, action)] += alpha * (reward + gamma * best_next - Q[(state, action)])
```

In plain terms: *the value of this move is what I got right now, plus how good the
square I landed on looks.* Because "how good the next square looks" is itself learned
the same way, value seeps backwards from the goal, one square per pass, until it
reaches the start.

| | | |
|---|---|---|
| `alpha` | 0.1 | how hard to nudge — small, so one lucky run cannot dominate |
| `gamma` | 0.9 | how much a future reward counts against an immediate one |
| `epsilon` | 0.2 | how often to ignore the table and try something random |

**Why `epsilon` matters.** An agent that always picks its current best move gets
stuck on the first route it stumbles into and never discovers a shorter one. Twenty
percent of moves are deliberately random, purely so it keeps finding out what it
does not know yet.

## What actually happens

Real output, running the training:

| Episode | Steps taken | |
|---:|---:|---|
| 0 | 150 | gave up at the step cap |
| 20 | 34 | reached the goal |
| 100 | 31 | reached the goal |
| 300 | 57 | reached the goal |
| 800 | 11 | reached the goal |
| 1999 | 11 | reached the goal |

The first run is pure flailing — it hits the 150-step cap without ever finding the
exit. By episode 20 it gets there, badly. The wobble at episode 300 is not a bug:
that is exploration, still trying random detours.

By the end, following the table greedily gives:

```
Path: [(0,0), (0,1), (0,2), (0,3), (0,4), (0,5), (1,5), (2,5), (3,5), (4,5), (5,5)]
Reached goal!
```

**10 moves — the shortest route that exists.** Across 20 runs with different random
seeds, it found the optimal path 20 out of 20 times.

## The two notebooks

**`maze_solving_agent_solved.ipynb`** — trains, solves, and animates the finished
agent walking the maze. Fixed `epsilon` at 0.2 throughout.

**`maze_solving_agent_steps.ipynb`** — the more interesting one. It snapshots the
agent's route at episodes 0, 20, 100, 300, 800 and 1999, then animates all six back
to back so you can watch the flailing turn into a straight line. This version also
decays `epsilon` from 1.0 down to 0.05, so it explores wildly at first and commits
to what it knows later.

Both render inline with `matplotlib.animation` — GitHub will not play them, so run
the notebook to see the animation.

## Running it

```bash
pip install numpy matplotlib jupyter
```

```bash
jupyter notebook maze_solving_agent_steps.ipynb
```

Run all cells. Training 2,000 episodes takes a couple of seconds.

There is no random seed set, so your numbers will differ slightly from the table
above. The final path should still be 10 moves.

## What this is and is not

This is **tabular** Q-learning — the agent memorises a value for every square it has
personally stood on. That works here because there are only 36 squares.

It does not generalise. Move one wall and the table is worthless; the agent has
learned *this* maze, not mazes. It has no idea what a wall is, no concept of
distance, and cannot look at a square it has never visited and guess anything about
it.

Getting past that means replacing the lookup table with a neural network that takes
the state as input and predicts values — Deep Q-Networks. That is the natural next
step, and the reason this small version is worth writing first: everything in DQN is
this same update rule, with the table swapped for something that can generalise.
