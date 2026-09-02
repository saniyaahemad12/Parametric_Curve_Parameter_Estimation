import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution

# --------------------------------------------------
# 1. Load observed data
# --------------------------------------------------

data = pd.read_csv("xy_data.csv")

x_obs = data["x"].values
y_obs = data["y"].values

print("Number of points:", len(x_obs))


# --------------------------------------------------
# 2. Parametric curve
# --------------------------------------------------

def curve(t, theta_deg, M, X):
    theta = np.radians(theta_deg)

    f = np.exp(M * np.abs(t)) * np.sin(0.3 * t)

    x = t * np.cos(theta) - f * np.sin(theta) + X
    y = 42 + t * np.sin(theta) + f * np.cos(theta)

    return x, y


# --------------------------------------------------
# 3. L1 distance from observed points to predicted curve
# --------------------------------------------------

# Uniformly sample t between 6 and 60
t_grid = np.linspace(6, 60, 5000)


def objective(params):

    theta, M, X = params

    x_pred, y_pred = curve(t_grid, theta, M, X)

    # Calculate nearest L1 distance for every observed point
    total_distance = 0.0

    # Process in chunks to avoid excessive memory usage
    chunk_size = 100

    for i in range(0, len(x_obs), chunk_size):

        xo = x_obs[i:i + chunk_size]
        yo = y_obs[i:i + chunk_size]

        distances = (
            np.abs(xo[:, None] - x_pred[None, :])
            +
            np.abs(yo[:, None] - y_pred[None, :])
        )

        nearest = np.min(distances, axis=1)

        total_distance += np.sum(nearest)

    return total_distance


# --------------------------------------------------
# 4. Optimize theta, M and X
# --------------------------------------------------

bounds = [
    (0.0001, 49.9999),     # theta
    (-0.049999, 0.049999), # M
    (0.0001, 99.9999)      # X
]

print("\nSearching for best parameters...")
print("This may take a few minutes...\n")

result = differential_evolution(
    objective,
    bounds,
    seed=42,
    popsize=15,
    maxiter=100,
    tol=1e-7,
    polish=True,
    workers=1
)


# --------------------------------------------------
# 5. Print result
# --------------------------------------------------

theta, M, X = result.x

print("\n==============================")
print("BEST FIT")
print("==============================")

print(f"theta = {theta:.8f} degrees")
print(f"M     = {M:.8f}")
print(f"X     = {X:.8f}")

print(f"\nL1 distance = {result.fun:.6f}")
print(f"Optimization success = {result.success}")
print(f"Message = {result.message}")


# --------------------------------------------------
# 6. Plot observed vs fitted curve
# --------------------------------------------------

x_pred, y_pred = curve(t_grid, theta, M, X)

plt.figure(figsize=(10, 7))

plt.scatter(
    x_obs,
    y_obs,
    s=8,
    alpha=0.5,
    label="Observed data"
)

plt.plot(
    x_pred,
    y_pred,
    linewidth=2,
    label="Fitted curve"
)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Observed Data vs Fitted Parametric Curve")
plt.legend()
plt.grid(True)

plt.savefig("fitted_curve.png", dpi=200)

plt.show()