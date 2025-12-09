import tkinter as tk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# -------------------------------
# Light Mode for Graphs
# -------------------------------
plt.style.use("default")  # Light theme for Matplotlib

# -------------------------------
# Multi-Armed Bandit Environment
# -------------------------------
class AdBandit:
    def __init__(self, ads, ctrs):
        self.ads = np.array(ads)
        self.ctrs = np.array(ctrs)
        self.K = len(ads)

    def pull(self, arm):
        return 1 if np.random.rand() < self.ctrs[arm] else 0

    def expected_rewards(self):
        return self.ctrs

# -------------------------------
# Algorithms
# -------------------------------
def epsilon_greedy(env, T, eps=0.1):
    K = env.K
    counts = np.zeros(K)
    values = np.zeros(K)
    rewards = np.zeros(T)
    chosen = np.zeros(T, dtype=int)
    for t in range(T):
        arm = np.random.randint(K) if np.random.rand() < eps else np.argmax(values)
        r = env.pull(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        rewards[t] = r
        chosen[t] = arm
    return rewards, chosen

def ucb1(env, T):
    K = env.K
    counts = np.ones(K)
    values = np.zeros(K)
    rewards = np.zeros(T)
    chosen = np.zeros(T, dtype=int)
    for t in range(T):
        ucb = values + np.sqrt((2 * np.log(t + 1)) / counts)
        arm = np.argmax(ucb)
        r = env.pull(arm)
        counts[arm] += 1
        values[arm] += (r - values[arm]) / counts[arm]
        rewards[t] = r
        chosen[t] = arm
    return rewards, chosen

def thompson_sampling(env, T):
    K = env.K
    alpha = np.ones(K)
    beta = np.ones(K)
    rewards = np.zeros(T)
    chosen = np.zeros(T, dtype=int)
    for t in range(T):
        theta = np.random.beta(alpha, beta)
        arm = np.argmax(theta)
        r = env.pull(arm)
        rewards[t] = r
        chosen[t] = arm
        alpha[arm] += r
        beta[arm] += (1 - r)
    return rewards, chosen

# -------------------------------
# GUI Application
# -------------------------------
class BanditApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ad Banner Multi-Armed Bandit")
        self.root.geometry("1380x900")
        self.root.configure(bg="#f7f7f7")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.splash_screen()

    # Splash Screen
    def splash_screen(self):
        self.clear()
        splash = tk.Frame(self.root, bg="#f7f7f7")
        splash.pack(fill="both", expand=True)

        tk.Label(
            splash, text="Ad Banner Multi-Armed Bandit",
            font=("Segoe UI", 44, "bold"),
            fg="#4b0082", bg="#f7f7f7"
        ).pack(expand=True)

        tk.Label(
            splash, text="ε-Greedy • UCB1 • Thompson Sampling",
            font=("Segoe UI", 22),
            fg="#555555", bg="#f7f7f7"
        ).pack()

        self.root.after(2000, self.run_simulation)

    # Run Simulation
    def run_simulation(self):
        data = pd.read_csv("ads_dataset.csv")
        ads = data["Ad"].tolist()
        ctrs = data["CTR"].tolist()
        T = 5000
        env = AdBandit(ads, ctrs)

        eg_r, eg_c = epsilon_greedy(env, T)
        ucb_r, ucb_c = ucb1(env, T)
        ts_r, ts_c = thompson_sampling(env, T)

        optimal = np.max(env.expected_rewards())
        regret_eg = np.cumsum(optimal - eg_r)
        regret_ucb = np.cumsum(optimal - ucb_r)
        regret_ts = np.cumsum(optimal - ts_r)

        self.results = {
            "eg": np.cumsum(eg_r),
            "ucb": np.cumsum(ucb_r),
            "ts": np.cumsum(ts_r),
            "reg_eg": regret_eg,
            "reg_ucb": regret_ucb,
            "reg_ts": regret_ts,
            "ads": ads,
            "expected": ctrs
        }
        self.show_results()

    # Show Results
    def show_results(self):
        self.clear()
        wrapper = tk.Frame(self.root, bg="#f7f7f7")
        wrapper.pack(fill="both", expand=True)
        wrapper.pack_propagate(False)  # Make sure frame doesn't shrink

        # ---------------- Graphs ----------------
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))  # Slightly bigger
        x = np.arange(len(self.results["eg"]))

        # Cumulative Clicks
        ax[0].plot(x, self.results["eg"], label="ε-Greedy", linewidth=2)
        ax[0].scatter(x[::200], self.results["eg"][::200], s=15)
        ax[0].plot(x, self.results["ucb"], label="UCB1", linewidth=2)
        ax[0].scatter(x[::200], self.results["ucb"][::200], s=15)
        ax[0].plot(x, self.results["ts"], label="Thompson Sampling", linewidth=2)
        ax[0].scatter(x[::200], self.results["ts"][::200], s=15)
        ax[0].set_title("Cumulative Clicks", fontsize=14)
        ax[0].grid(True)
        ax[0].legend(fontsize=10)

        best_idx = np.argmax(self.results["expected"])
        ax[0].axvline(best_idx * 1000, color="#ff69b4", linestyle="--", linewidth=2)

        # Cumulative Regret
        ax[1].plot(x, self.results["reg_eg"], label="ε-Greedy", linewidth=2)
        ax[1].scatter(x[::200], self.results["reg_eg"][::200], s=15)
        ax[1].plot(x, self.results["reg_ucb"], label="UCB1", linewidth=2)
        ax[1].scatter(x[::200], self.results["reg_ucb"][::200], s=15)
        ax[1].plot(x, self.results["reg_ts"], label="Thompson Sampling", linewidth=2)
        ax[1].scatter(x[::200], self.results["reg_ts"][::200], s=15)
        ax[1].set_title("Cumulative Regret", fontsize=14)
        ax[1].grid(True)
        ax[1].legend(fontsize=10)

        # Centering Canvas
        canvas = FigureCanvasTkAgg(fig, wrapper)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(expand=True)
        canvas_widget.place(relx=0.5, rely=0.45, anchor="center")  # Center in frame

        # ---------------- BEST AD Label ----------------
        tk.Label(
            wrapper,
            text=f"BEST AD → {self.results['ads'][best_idx]} (CTR = {round(self.results['expected'][best_idx],3)})",
            font=("Segoe UI", 18, "bold"),
            fg="#4b0082",
            bg="#f7f7f7"
        ).place(relx=0.5, rely=0.88, anchor="center")  # Center at bottom

    # Clear Window
    def clear(self):
        for w in self.root.winfo_children():
            w.destroy()

# -------------------------------
# Run App
# -------------------------------
root = tk.Tk()
app = BanditApp(root)
root.mainloop()
