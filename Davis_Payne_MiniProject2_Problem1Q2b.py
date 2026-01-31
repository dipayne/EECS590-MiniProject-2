"""
Windy Chasm — Problem 1, Question 2(b)
Continuous-approximation solver and visualization on [0, 19] x [0, 6] using:
  - horizontal step dx = 0.05  -> 381 columns
  - vertical step   dy = 1     -> 7 rows

This script includes:
  1) Tabular MDP model (x stored as integer index to avoid float key issues)
  2) Value iteration to compute optimal value function and optimal policy
  3) Static optimal policy plot (PNG)
  4) Static value function plot (PNG)
  5) Pygame animation: agent follows the optimal policy inside the same stochastic environment

Dependencies:
  - numpy
  - matplotlib
  - pygame

Install pygame if needed:
  pip install pygame
"""

from __future__ import annotations

from collections import defaultdict
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pygame

# ------------------------------------------------------------
# Types and constants
# ------------------------------------------------------------
State = Union[Tuple[int, int], str]  # (x_idx, j) or terminal label
ACTIONS = ["forward", "left", "right"]


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value into [lo, hi]."""
    return max(lo, min(hi, x))


@dataclass(frozen=True)
class WindyChasmConfig:
    # Discretization for Problem 1, Question 2(b)
    dx: float = 0.05  # horizontal step size (refined)
    dy: float = 1.0   # vertical step size (kept coarse)

    # Continuous rectangle bounds
    i_min: float = 0.0
    i_max: float = 19.0
    j_min: int = 0
    j_max: int = 6

    # Rewards
    R_goal: float = 20.0
    r_crash: float = 10.0

    # Discount
    gamma: float = 0.95

    # Wind baseline parameter (interpreted as B = p(3) for p(j)=B^{E(j)})
    B_center: float = 0.30

    # Value iteration
    tol: float = 1e-8
    max_iter: int = 20000

    # Output/plots
    output_prefix: str = "windy_chasm_2b"
    show_plots: bool = True

    # Simulation summary (optional)
    sim_episodes: int = 200
    sim_max_steps: int = 5000

    # Animation settings
    animate: bool = True
    anim_fps: int = 30
    anim_max_steps: int = 5000
    cell_w: int = 4   # pixel width per micro-column (381*4 = 1524 px wide)
    cell_h: int = 60  # pixel height per row (7*60 = 420 px tall)
    margin: int = 10


class WindyChasmMDPContinuousApprox:
    """
    Continuous-approximation MDP for Windy Chasm using:
      - horizontal index x_idx representing i = x_idx * dx
      - vertical coordinate j in {0,1,...,6}

    Terminals:
      - GOAL: reaching final column (x_idx == N-1)
      - CRASH: entering boundary rows j==0 or j==6

    Wind model:
      - p(j) = B^{E(j)} (coarse-step)
      - p_micro(j) = 1 - (1 - p(j))^{dx} (micro-step conversion)
      - After action:
          ±1 with prob p_micro split 50/50
          ±2 with prob (1-p_micro)*p_micro^2 split 50/50
          stay otherwise
    """

    GOAL = "GOAL"
    CRASH = "CRASH"

    def __init__(self, cfg: WindyChasmConfig, seed: int = 0):
        self.cfg = cfg
        self.rng = random.Random(seed)

        # 0..19 inclusive in dx steps => N columns
        self.n_cols = int(round((cfg.i_max - cfg.i_min) / cfg.dx)) + 1  # 381 for dx=0.05
        self.x_vals = list(range(self.n_cols))
        self.j_vals = list(range(cfg.j_min, cfg.j_max + 1))  # 0..6

        self.start_state: State = (0, 3)
        self.grid_states: List[State] = [(x, j) for x in self.x_vals for j in self.j_vals]
        self.states: List[State] = self.grid_states + [self.GOAL, self.CRASH]

    def x_to_i(self, x_idx: int) -> float:
        """Convert horizontal index to physical i coordinate."""
        return float(x_idx) * self.cfg.dx

    # ---------------------------
    # Terminal checks
    # ---------------------------
    def is_goal_cell(self, s: State) -> bool:
        return isinstance(s, tuple) and (s[0] >= self.n_cols - 1)

    def is_crash_cell(self, s: State) -> bool:
        return isinstance(s, tuple) and (s[1] <= self.cfg.j_min or s[1] >= self.cfg.j_max)

    # ---------------------------
    # Wind model
    # ---------------------------
    def E(self, j: int) -> float:
        """Centerline weighting term."""
        return 1.0 / (1.0 + (j - 3) ** 2)

    def p_discrete(self, j: int) -> float:
        """Coarse-step wind probability: p(j) = B^{E(j)}."""
        B = float(self.cfg.B_center)
        return clamp(B ** self.E(j), 0.0, 1.0)

    def p_micro(self, j: int) -> float:
        """
        Convert coarse probability to micro-step probability:
          p_micro = 1 - (1 - p)^{dx}
        """
        p = self.p_discrete(j)
        return clamp(1.0 - (1.0 - p) ** self.cfg.dx, 0.0, 1.0)

    # ---------------------------
    # Dynamics
    # ---------------------------
    def deterministic_move(self, s: State, a: str) -> State:
        if s in [self.GOAL, self.CRASH]:
            return s

        x, j = s  # type: ignore[misc]

        if a == "forward":
            return (min(x + 1, self.n_cols - 1), j)
        if a == "left":
            return (x, j - 1)
        if a == "right":
            return (x, j + 1)

        raise ValueError(f"Unknown action: {a}")

    def wind_distribution(self, s_after_action: State) -> Dict[State, float]:
        if s_after_action in [self.GOAL, self.CRASH]:
            return {s_after_action: 1.0}
        if self.is_goal_cell(s_after_action):
            return {self.GOAL: 1.0}
        if self.is_crash_cell(s_after_action):
            return {self.CRASH: 1.0}

        x, j = s_after_action  # type: ignore[misc]
        p = self.p_micro(j)

        dist = defaultdict(float)

        # ±1 with probability p
        dist[(x, j + 1)] += p / 2.0
        dist[(x, j - 1)] += p / 2.0

        # ±2 with probability (1-p)*p^2
        p2 = (1.0 - p) * (p ** 2)
        dist[(x, j + 2)] += p2 / 2.0
        dist[(x, j - 2)] += p2 / 2.0

        # Stay otherwise
        p_stay = 1.0 - (p + p2)
        dist[(x, j)] += max(0.0, p_stay)

        # Map to terminals
        final = defaultdict(float)
        for s2, prob in dist.items():
            if self.is_goal_cell(s2):
                final[self.GOAL] += prob
            elif self.is_crash_cell(s2):
                final[self.CRASH] += prob
            else:
                final[s2] += prob

        # Normalize
        total = sum(final.values())
        for k in list(final.keys()):
            final[k] /= total

        return dict(final)

    def transitions(self, s: State, a: str) -> Dict[State, float]:
        if s == self.GOAL:
            return {self.GOAL: 1.0}
        if s == self.CRASH:
            return {self.CRASH: 1.0}

        s_after = self.deterministic_move(s, a)
        return self.wind_distribution(s_after)

    # ---------------------------
    # Rewards
    # ---------------------------
    def reward(self, s: State, a: str, s_next: State) -> float:
        if s_next == self.GOAL:
            return float(self.cfg.R_goal)
        if s_next == self.CRASH:
            return -float(self.cfg.r_crash)

        # Micro-step penalty: approximately -1 per unit horizontal distance
        return -float(self.cfg.dx)

    # ---------------------------
    # Planning: value iteration
    # ---------------------------
    def q_value(self, s: State, a: str, V: Dict[State, float]) -> float:
        return sum(
            p * (self.reward(s, a, s2) + self.cfg.gamma * V[s2])
            for s2, p in self.transitions(s, a).items()
        )

    def value_iteration(self) -> Tuple[Dict[State, Dict[str, float]], Dict[State, float]]:
        V: Dict[State, float] = {s: 0.0 for s in self.states}

        for _ in range(self.cfg.max_iter):
            delta = 0.0
            V_new = V.copy()

            for s in self.states:
                if s in [self.GOAL, self.CRASH]:
                    continue
                V_new[s] = max(self.q_value(s, a, V) for a in ACTIONS)
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            if delta < self.cfg.tol:
                break

        # Greedy deterministic policy
        pi: Dict[State, Dict[str, float]] = {s: {a: 0.0 for a in ACTIONS} for s in self.states}
        for s in self.states:
            if s in [self.GOAL, self.CRASH]:
                continue
            best_a = max(ACTIONS, key=lambda a: self.q_value(s, a, V))
            for a in ACTIONS:
                pi[s][a] = 1.0 if a == best_a else 0.0

        return pi, V


# ------------------------------------------------------------
# Plotting
# ------------------------------------------------------------
def greedy_action(pi_s: Dict[str, float]) -> str:
    return max(pi_s, key=pi_s.get)


def plot_optimal_policy(mdp: WindyChasmMDPContinuousApprox, pi: Dict[State, Dict[str, float]], filename: str, show: bool):
    action_to_int = {"left": -1, "forward": 0, "right": 1}
    grid = np.zeros((len(mdp.j_vals), len(mdp.x_vals)), dtype=float)

    for x in mdp.x_vals:
        for j in mdp.j_vals:
            s = (x, j)
            if s not in pi:
                continue
            a = greedy_action(pi[s])
            grid[len(mdp.j_vals) - 1 - j, x] = action_to_int[a]

    plt.figure(figsize=(12, 3))
    plt.imshow(grid, aspect="auto")
    plt.colorbar(label="Greedy action: -1=left(down), 0=forward, +1=right(up)")
    plt.xlabel(f"Horizontal index (i = x * dx, dx={mdp.cfg.dx})")
    plt.ylabel("Vertical position j (0 bottom to 6 top)")
    plt.title("Optimal Policy (Problem 1, Q2b continuous approximation)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


def plot_value_function(mdp: WindyChasmMDPContinuousApprox, V: Dict[State, float], filename: str, show: bool):
    grid = np.full((len(mdp.j_vals), len(mdp.x_vals)), np.nan, dtype=float)

    for x in mdp.x_vals:
        for j in mdp.j_vals:
            s = (x, j)
            if s in V:
                grid[len(mdp.j_vals) - 1 - j, x] = V[s]

    plt.figure(figsize=(12, 3))
    plt.imshow(grid, aspect="auto")
    plt.colorbar(label="V*(state)")
    plt.xlabel(f"Horizontal index (i = x * dx, dx={mdp.cfg.dx})")
    plt.ylabel("Vertical position j (0 bottom to 6 top)")
    plt.title("Optimal Value Function (Problem 1, Q2b continuous approximation)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    if show:
        plt.show()
    plt.close()


# ------------------------------------------------------------
# Simulation stats
# ------------------------------------------------------------
def simulate_policy(mdp: WindyChasmMDPContinuousApprox, pi: Dict[State, Dict[str, float]], n_episodes: int, max_steps: int, seed: int = 0) -> Dict[str, float]:
    rng = random.Random(seed)
    goals = 0
    crashes = 0
    returns = []
    steps_list = []

    for _ in range(n_episodes):
        s: State = mdp.start_state
        total_r = 0.0
        steps = 0

        while steps < max_steps:
            steps += 1
            if s in [mdp.GOAL, mdp.CRASH]:
                break

            a = greedy_action(pi[s])
            dist = mdp.transitions(s, a)

            u = rng.random()
            cum = 0.0
            s_next: Optional[State] = None
            for st, p in dist.items():
                cum += p
                if u <= cum:
                    s_next = st
                    break
            if s_next is None:
                s_next = list(dist.keys())[-1]

            total_r += mdp.reward(s, a, s_next)
            s = s_next

            if s == mdp.GOAL or s == mdp.CRASH:
                break

        if s == mdp.GOAL:
            goals += 1
        elif s == mdp.CRASH:
            crashes += 1

        returns.append(total_r)
        steps_list.append(steps)

    return {
        "goal_rate": goals / n_episodes,
        "crash_rate": crashes / n_episodes,
        "avg_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "avg_steps": float(np.mean(steps_list)),
    }


# ------------------------------------------------------------
# Pygame animation
# ------------------------------------------------------------
class WindyChasmAnimator:
    """
    Pygame animation of one episode under the greedy optimal policy.
    Rendering uses:
      - width  = n_cols * cell_w
      - height = 7 * cell_h
    """

    def __init__(self, mdp: WindyChasmMDPContinuousApprox, pi: Dict[State, Dict[str, float]]):
        self.mdp = mdp
        self.pi = pi

        cfg = mdp.cfg
        self.cell_w = cfg.cell_w
        self.cell_h = cfg.cell_h
        self.margin = cfg.margin

        self.w = mdp.n_cols * self.cell_w + 2 * self.margin
        self.h = len(mdp.j_vals) * self.cell_h + 2 * self.margin

        self.screen = None
        self.clock = None
        self.font = None

    def run(self, seed: int = 123, max_steps: int = 5000, fps: int = 30):
        pygame.init()
        self.screen = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Windy Chasm 2(b) — Optimal Policy Animation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 18)

        # Use a separate RNG for episode stochasticity
        episode_rng = random.Random(seed)

        s: State = self.mdp.start_state
        total_r = 0.0
        steps = 0
        done_label = None  # "GOAL" or "CRASH"

        running = True
        while running:
            self.clock.tick(fps)

            # Handle quit events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            if done_label is None and steps < max_steps:
                steps += 1

                # Choose greedy action from policy
                a = greedy_action(self.pi[s])

                # Sample from transition distribution
                dist = self.mdp.transitions(s, a)
                u = episode_rng.random()
                cum = 0.0
                s_next: Optional[State] = None
                for st, p in dist.items():
                    cum += p
                    if u <= cum:
                        s_next = st
                        break
                if s_next is None:
                    s_next = list(dist.keys())[-1]

                total_r += self.mdp.reward(s, a, s_next)

                # Update physical state if not terminal label
                if s_next == self.mdp.GOAL:
                    done_label = "GOAL"
                elif s_next == self.mdp.CRASH:
                    done_label = "CRASH"
                else:
                    s = s_next

            self.render(s, steps, total_r, done_label)
            pygame.display.flip()

        pygame.quit()

    def render(self, s: State, steps: int, total_r: float, done_label: Optional[str]):
        assert self.screen is not None
        assert self.font is not None

        cfg = self.mdp.cfg

        # Background
        self.screen.fill((255, 255, 255))

        # Draw grid cells
        for x in range(self.mdp.n_cols):
            for j in self.mdp.j_vals:
                px = self.margin + x * self.cell_w
                py = self.margin + (cfg.j_max - j) * self.cell_h  # flip vertically

                # Crash rows (j=0 and j=6)
                if j <= cfg.j_min or j >= cfg.j_max:
                    color = (235, 235, 235)
                # Goal column (x == last)
                elif x == self.mdp.n_cols - 1:
                    color = (210, 255, 210)
                else:
                    color = (245, 245, 245)

                pygame.draw.rect(self.screen, color, (px, py, self.cell_w, self.cell_h))

        # Draw start marker
        sx, sj = 0, 3
        spx = self.margin + sx * self.cell_w + self.cell_w // 2
        spy = self.margin + (cfg.j_max - sj) * self.cell_h + self.cell_h // 2
        pygame.draw.circle(self.screen, (0, 160, 0), (spx, spy), max(3, self.cell_h // 10))

        # Draw agent
        if isinstance(s, tuple):
            x, j = s
            ax = self.margin + x * self.cell_w + self.cell_w // 2
            ay = self.margin + (cfg.j_max - j) * self.cell_h + self.cell_h // 2
            pygame.draw.circle(self.screen, (60, 90, 220), (ax, ay), max(5, self.cell_h // 4))

        # HUD text
        i_coord = self.mdp.x_to_i(s[0]) if isinstance(s, tuple) else float("nan")  # type: ignore[index]
        status = f"steps={steps}  return={total_r:.3f}  i={i_coord:.2f}"
        if done_label is not None:
            status += f"  TERMINAL={done_label}"
        text_surf = self.font.render(status, True, (0, 0, 0))
        self.screen.blit(text_surf, (10, 10))


# ------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------
def main() -> None:
    cfg = WindyChasmConfig(
        dx=0.05,
        dy=1.0,
        R_goal=20.0,
        r_crash=10.0,
        gamma=0.95,
        B_center=0.30,  # increase if reaching the goal too easily
        tol=1e-8,
        max_iter=20000,
        output_prefix="windy_chasm_2b",
        show_plots=True,
        sim_episodes=200,
        sim_max_steps=5000,
        animate=True,
        anim_fps=30,
        anim_max_steps=5000,
        cell_w=4,
        cell_h=60,
        margin=10,
    )

    mdp = WindyChasmMDPContinuousApprox(cfg, seed=42)
    pi_opt, V_opt = mdp.value_iteration()

    start_val = V_opt[mdp.start_state]
    print(f"V*(start) = {start_val:.6f}")

    # Static plots for report
    policy_path = f"{cfg.output_prefix}_policy.png"
    value_path = f"{cfg.output_prefix}_value.png"
    plot_optimal_policy(mdp, pi_opt, filename=policy_path, show=cfg.show_plots)
    plot_value_function(mdp, V_opt, filename=value_path, show=cfg.show_plots)
    print(f"Saved policy plot to: {policy_path}")
    print(f"Saved value plot to:  {value_path}")

    # Optional simulation summary
    stats = simulate_policy(mdp, pi_opt, n_episodes=cfg.sim_episodes, max_steps=cfg.sim_max_steps, seed=123)
    print("Simulation summary (greedy policy):")
    print(f"  goal_rate   = {stats['goal_rate']:.3f}")
    print(f"  crash_rate  = {stats['crash_rate']:.3f}")
    print(f"  avg_return  = {stats['avg_return']:.3f} (std {stats['std_return']:.3f})")
    print(f"  avg_steps   = {stats['avg_steps']:.1f}")

    # Pygame animation
    if cfg.animate:
        animator = WindyChasmAnimator(mdp, pi_opt)
        animator.run(seed=777, max_steps=cfg.anim_max_steps, fps=cfg.anim_fps)


if __name__ == "__main__":
    main()
