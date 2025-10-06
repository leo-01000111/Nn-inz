import optuna
import numpy as np
import matplotlib.pyplot as plt
import optuna.visualization as vis

def evaluate_function(x, a, b):
    return (x - a)**2 + np.sin(3*x) + b * np.cos(5*x) + 0.1 * b**2


a_min = -5
a_max = 5
b_min = -6
b_max = 6


def objective(trial):
    a = trial.suggest_float("a", a_min, a_max)
    b = trial.suggest_float("b", b_min, b_max)
    xs = np.linspace(-5, 5, 2000)
    vals = evaluate_function(xs, a, b)
    return np.min(vals)



#wykresy, jak zwykle, pisze za mnie gpt, mam nadzieję że do wybaczenia
def visualize_function_samples(best_params, n_samples=10, zoom=2.0):
    a_best = best_params["a"]
    b_best = best_params["b"]

    # Compute best x for the best a and b (true minimum in the grid)
    xs_full = np.linspace(a_min - 1, a_max + 1, 2000)
    ys_best_full = evaluate_function(xs_full, a_best, b_best)
    x_min_best = xs_full[np.argmin(ys_best_full)]
    y_min_best = np.min(ys_best_full)

    # Center xs around the best minimum
    xs = np.linspace(x_min_best - zoom, x_min_best + zoom, 1000)

    plt.figure(figsize=(10, 6))
    plt.title(f"Function samples near best (a={a_best:.2f}, b={b_best:.2f})", fontsize=14)
    plt.xlabel("x")
    plt.ylabel("f(x, a, b)")

    # Plot random nearby functions
    for _ in range(n_samples):
        a = np.random.normal(a_best, 0.9)
        b = np.random.normal(b_best, 0.6)
        ys = evaluate_function(xs, a, b)
        plt.plot(xs, ys, color="gray", alpha=0.35, linewidth=1)
        x_min = xs[np.argmin(ys)]
        y_min = np.min(ys)
        plt.scatter(x_min, y_min, color="black", s=15, alpha=0.5)

    # Plot best function
    ys_best = evaluate_function(xs, a_best, b_best)
    plt.plot(xs, ys_best, color="red", linewidth=2.5, label="Best Optuna function")
    plt.scatter(x_min_best, y_min_best, color="lime", s=60, edgecolor="black", zorder=5, label="Best minimum")

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=500)

    print("Best parameters:", study.best_params)
    print("Best value:", study.best_value)

    vis.plot_optimization_history(study).show()

    visualize_function_samples(study.best_params)
