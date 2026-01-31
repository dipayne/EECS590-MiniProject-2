"""
Windy Chasm (MDP solver + Gymnasium animation)
"""

from collections import defaultdict
import random
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame

# ---------------------------
# Shared constants
# ---------------------------
ACTIONS = ["forward", "left", "right"]  # map to gym actions 0,1,2


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


# ------------------------------------------------------------
# Mini Project 1 style MRP (used for policy evaluation)
# ------------------------------------------------------------
class MarkovRewardProcess:
    def __init__(self, states, rewards, transitions, gamma=0.9):
        self.states = states
        self.R = rewards          # dict: R[s] = E[R_{t+1} | S_t=s]
        self.P = transitions      # dict: P[s][s']
        self.gamma = gamma

    def value_iteration(self, tol=1e-8, max_iter=10000):
        """
        MRP Bellman expectation backup:
          V(s) = R(s) + gamma * sum_{s'} P(s'|s) V(s')
        """
        V = {s: 0.0 for s in self.states}

        for _ in range(max_iter):
            delta = 0.0
            V_new = V.copy()

            for s in self.states:
                V_new[s] = self.R[s] + self.gamma * sum(
                    self.P[s].get(s_next, 0.0) * V[s_next] for s_next in self.states
                )
                delta = max(delta, abs(V_new[s] - V[s]))

            V = V_new
            if delta < tol:
                break

        return V


# ------------------------------------------------------------
# Problem 1 MDP
# ------------------------------------------------------------
class WindyChasmMDP:
    def __init__(self, p_center=0.3, R_goal=20.0, r_crash=10.0, gamma=0.95, seed=0):
        self.p_center = p_center
        self.R_goal = R_goal
        self.r_crash = r_crash
        self.gamma = gamma
        self.rng = random.Random(seed)

        self.grid_states = [(i, j) for i in range(20) for j in range(7)]
        self.GOAL = "GOAL"
        self.CRASH = "CRASH"
        self.states = self.grid_states + [self.GOAL, self.CRASH]
        self.start_state = (0, 3)

    def is_goal_cell(self, s):
        return isinstance(s, tuple) and s[0] == 19 and 0 <= s[1] <= 6

    def is_crash_cell(self, s):
        return isinstance(s, tuple) and (s[1] <= 0 or s[1] >= 6)

    def E(self, j):
        return 1.0 / (1.0 + (j - 3) ** 2)

    def p_eff(self, j):
        val = self.p_center / self.E(j)
        return clamp(val, 0.0, 1.0)

    def deterministic_move(self, s, a):
        if s in [self.GOAL, self.CRASH]:
            return s

        i, j = s
        if a == "forward":
            return (min(i + 1, 19), j)
        elif a == "left":
            return (i, j - 1)
        elif a == "right":
            return (i, j + 1)
        else:
            raise ValueError(f"Unknown action: {a}")

    def wind_distribution(self, s_after_action):
        """
        Wind after deterministic action:
          ±1 with prob p (50/50)
          ±2 with prob (1-p)*p^2 (50/50)
          stay with prob (1-p)*(1-p^2)

        For j != 3 use p(j)=B/E(j), clamped.
        """
        if s_after_action in [self.GOAL, self.CRASH]:
            return {s_after_action: 1.0}

        if self.is_goal_cell(s_after_action):
            return {self.GOAL: 1.0}
        if self.is_crash_cell(s_after_action):
            return {self.CRASH: 1.0}

        i, j = s_after_action
        p = self.p_center if j == 3 else self.p_eff(j)

        dist = defaultdict(float)

        # ±1 with prob p
        dist[(i, j + 1)] += p / 2.0
        dist[(i, j - 1)] += p / 2.0

        # ±2 with prob (1-p)*p^2
        p2 = (1.0 - p) * (p ** 2)
        dist[(i, j + 2)] += p2 / 2.0
        dist[(i, j - 2)] += p2 / 2.0

        # stay otherwise
        p_stay = (1.0 - p) * (1.0 - (p ** 2))
        dist[(i, j)] += p_stay

        # Map to terminals if crash/goal
        final = defaultdict(float)
        for s2, prob in dist.items():
            if self.is_goal_cell(s2):
                final[self.GOAL] += prob
            elif self.is_crash_cell(s2):
                final[self.CRASH] += prob
            else:
                final[s2] += prob

        total = sum(final.values())
        for k in list(final.keys()):
            final[k] /= total

        return dict(final)

    def transitions(self, s, a):
        if s == self.GOAL:
            return {self.GOAL: 1.0}
        if s == self.CRASH:
            return {self.CRASH: 1.0}
        s_after = self.deterministic_move(s, a)
        return self.wind_distribution(s_after)

    def reward(self, s, a, s_next):
        if s_next == self.GOAL:
            return self.R_goal
        if s_next == self.CRASH:
            return -self.r_crash
        return -1.0

    def q_value(self, s, a, V):
        return sum(
            p * (self.reward(s, a, s2) + self.gamma * V[s2])
            for s2, p in self.transitions(s, a).items()
        )

    def induced_mrp(self, pi):
        R_pi = {}
        P_pi = {}

        for s in self.states:
            if s in [self.GOAL, self.CRASH]:
                R_pi[s] = 0.0
                P_pi[s] = {s: 1.0}
                continue

            R_pi[s] = 0.0
            P_pi[s] = defaultdict(float)

            for a, pa in pi[s].items():
                R_pi[s] += pa * sum(
                    p * self.reward(s, a, s2) for s2, p in self.transitions(s, a).items()
                )
                for s2, p in self.transitions(s, a).items():
                    P_pi[s][s2] += pa * p

            total = sum(P_pi[s].values())
            if total > 0:
                for s2 in list(P_pi[s].keys()):
                    P_pi[s][s2] /= total
            P_pi[s] = dict(P_pi[s])

        return MarkovRewardProcess(self.states, R_pi, P_pi, gamma=self.gamma)

    def value_iteration(self, tol=1e-8, max_iter=20000):
        V = {s: 0.0 for s in self.states}
        for _ in range(max_iter):
            delta = 0.0
            V_new = V.copy()
            for s in self.states:
                if s in [self.GOAL, self.CRASH]:
                    continue
                V_new[s] = max(self.q_value(s, a, V) for a in ACTIONS)
                delta = max(delta, abs(V_new[s] - V[s]))
            V = V_new
            if delta < tol:
                break

        pi = {s: {a: 0.0 for a in ACTIONS} for s in self.states}
        for s in self.states:
            if s in [self.GOAL, self.CRASH]:
                continue
            best_a = max(ACTIONS, key=lambda a: self.q_value(s, a, V))
            for a in ACTIONS:
                pi[s][a] = 1.0 if a == best_a else 0.0

        return pi, V


# ------------------------------------------------------------
# Gymnasium Env (renders + plays optimal policy)
# ------------------------------------------------------------
class WindyChasmGymEnv(gym.Env):
    """
    Gymnasium wrapper that uses the SAME windy dynamics as our MDP,
    but returns observations as (i,j) and renders live with pygame.
    """
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, mdp: WindyChasmMDP, render_mode="human", cell_size=40, max_steps=300):
        super().__init__()
        self.mdp = mdp
        self.render_mode = render_mode
        self.cell_size = cell_size
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(3)                  # 0,1,2
        self.observation_space = spaces.MultiDiscrete([20, 7])  # i in [0..19], j in [0..6]

        self.s = mdp.start_state
        self.t = 0

        # diagnostics
        self.last_terminal_type = None  # "GOAL" | "CRASH" | None

        # pygame
        self._screen = None
        self._clock = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.mdp.rng.seed(seed)
        self.s = self.mdp.start_state
        self.t = 0
        self.last_terminal_type = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.s, dtype=np.int64), {}

    def step(self, action):
        self.t += 1
        a_str = ACTIONS[int(action)]

        # Sample next state from mdp.transitions
        dist = self.mdp.transitions(self.s, a_str)

        r = self.mdp.rng.random()
        cum = 0.0
        s_next = None
        for st, p in dist.items():
            cum += p
            if r <= cum:
                s_next = st
                break
        if s_next is None:
            s_next = list(dist.keys())[-1]

        reward = self.mdp.reward(self.s, a_str, s_next)

        terminated = (s_next == self.mdp.GOAL) or (s_next == self.mdp.CRASH)
        truncated = (self.t >= self.max_steps)

        # record terminal type for diagnostics
        if s_next == self.mdp.GOAL:
            self.last_terminal_type = "GOAL"
        elif s_next == self.mdp.CRASH:
            self.last_terminal_type = "CRASH"
        else:
            self.last_terminal_type = None

        # Update position:
        # - If physical tuple, move there
        # - If terminal string, KEEP last physical position (so render doesn't crash)
        if isinstance(s_next, tuple):
            self.s = s_next

        if self.render_mode == "human":
            self.render()

        return np.array(self.s, dtype=np.int64), reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != "human":
            return None

        if self._screen is None:
            pygame.init()
            w = 20 * self.cell_size
            h = 7 * self.cell_size
            self._screen = pygame.display.set_mode((w, h))
            pygame.display.set_caption("Windy Chasm (Optimal Policy Animation)")
            self._clock = pygame.time.Clock()

        # allow window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None

        cs = self.cell_size
        self._screen.fill((255, 255, 255))

        # draw grid with crash rows and goal column
        for i in range(20):
            for j in range(7):
                x = i * cs
                y = (6 - j) * cs  # flip so j=6 is top

                if j <= 0 or j >= 6:
                    color = (230, 230, 230)      # crash rows
                elif i == 19:
                    color = (210, 255, 210)      # goal column
                else:
                    color = (245, 245, 245)

                pygame.draw.rect(self._screen, color, (x, y, cs, cs))
                pygame.draw.rect(self._screen, (0, 0, 0), (x, y, cs, cs), 1)

        # mark start
        sx, sy = 0 * cs + cs // 2, (6 - 3) * cs + cs // 2
        pygame.draw.circle(self._screen, (0, 160, 0), (sx, sy), cs // 6)

        # draw agent
        i, j = self.s
        ax = i * cs + cs // 2
        ay = (6 - j) * cs + cs // 2
        pygame.draw.circle(self._screen, (60, 90, 220), (ax, ay), cs // 3)

        pygame.display.flip()
        self._clock.tick(self.metadata["render_fps"])

    def close(self):
        if self._screen is not None:
            pygame.quit()
            self._screen = None
            self._clock = None


# ------------------------------------------------------------
# Main: compute optimal policy and animate it in Gymnasium
# ------------------------------------------------------------
if __name__ == "__main__":
    # ====== knobs you can change ======
    DEBUG_END_REASON = True     # prints why the episode stopped
    N_EPISODES = 1              # set to e.g. 50 to see goal/crash frequency
    RENDER = True               # set False to run fast without pygame window
    MAX_STEPS = 300             # if you see lots of truncation, increase this
    # ==================================

    # 1) Solve MDP (planning)
    mdp = WindyChasmMDP(p_center=0.3, R_goal=20.0, r_crash=10.0, gamma=0.95, seed=42)
    pi_opt, V_opt = mdp.value_iteration()
    print("Optimal V(start) =", V_opt[mdp.start_state])

    # 2) Create Gym env
    env = WindyChasmGymEnv(
        mdp,
        render_mode="human" if RENDER else None,
        cell_size=40,
        max_steps=MAX_STEPS
    )

    goals = 0
    crashes = 0
    truncs = 0
    returns = []

    for ep in range(1, N_EPISODES + 1):
        obs, _ = env.reset(seed=42 + ep)
        total_reward = 0.0
        steps = 0

        while True:
            steps += 1
            i, j = int(obs[0]), int(obs[1])

            # choose greedy action from optimal policy dict
            a_str = max(pi_opt[(i, j)], key=pi_opt[(i, j)].get)
            action = ACTIONS.index(a_str)

            obs, r, terminated, truncated, _ = env.step(action)
            total_reward += r

            if terminated or truncated:
                # end reason diagnostics
                if terminated:
                    if env.last_terminal_type == "GOAL":
                        goals += 1
                    elif env.last_terminal_type == "CRASH":
                        crashes += 1
                if truncated:
                    truncs += 1

                if DEBUG_END_REASON:
                    end_reason = "TRUNCATED (time limit)" if truncated else f"TERMINATED ({env.last_terminal_type})"
                    last_pos = (int(obs[0]), int(obs[1]))
                    print(
                        f"[Episode {ep}] ended at step {steps}: {end_reason} | "
                        f"last_pos={last_pos} | return={total_reward}"
                    )
                break

        returns.append(total_reward)

    env.close()

    if N_EPISODES == 1:
        print("Episode finished. Total reward:", returns[0])
    else:
        avg_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        print("\nSummary over", N_EPISODES, "episodes")
        print("  Goals:", goals)
        print("  Crashes:", crashes)
        print("  Truncations:", truncs)
        print(f"  Avg return: {avg_return:.3f}  (std {std_return:.3f})")
