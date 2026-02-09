"""
Vanilla PINN Example 4.1.2 (1D Heat Equation PDE) - JAX Implementation.

"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- 1. Data: Collocation, Initial, and Boundary Points ---
# Define physical parameters and domain
alpha = 0.01  # Thermal diffusivity
x_min, x_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0


# Initial condition (IC) function: u(x, 0) = sin(pi*x)
def u_initial(x):
    return np.sin(np.pi * x)


# Generate collocation points (for PDE residual)
key = jax.random.PRNGKey(0)
num_collocation = 5000

key, subkey = jax.random.split(key)
x_collocation = jax.random.uniform(subkey, (num_collocation, 1)) * (x_max - x_min) + x_min
key, subkey = jax.random.split(key)
t_collocation = jax.random.uniform(subkey, (num_collocation, 1)) * (t_max - t_min) + t_min
collocation_points = jnp.concatenate([x_collocation, t_collocation], axis=1)

# Generate initial condition points (IC)
num_ic = 100
key, subkey = jax.random.split(key)
x_ic = jax.random.uniform(subkey, (num_ic, 1)) * (x_max - x_min) + x_min
t_ic = jnp.zeros((num_ic, 1))
ic_points = jnp.concatenate([x_ic, t_ic], axis=1)
u_ic = jnp.array(u_initial(np.array(x_ic)))

# Generate boundary condition points (BC)
num_bc = 100
key, subkey = jax.random.split(key)
t_bc = jax.random.uniform(subkey, (num_bc, 1)) * (t_max - t_min) + t_min
x_bc_left = jnp.zeros((num_bc, 1))
x_bc_right = jnp.ones((num_bc, 1))
bc_points_left = jnp.concatenate([x_bc_left, t_bc], axis=1)
bc_points_right = jnp.concatenate([x_bc_right, t_bc], axis=1)
u_bc = jnp.zeros((2 * num_bc, 1))


# --- 2. Machine Learning Model: Neural Network ---
# This network will approximate the solution u(x, t)

class PINN(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Linear(2, 32, key=keys[0]),  # 2 inputs: x and t
            eqx.nn.Linear(32, 32, key=keys[1]),
            eqx.nn.Linear(32, 32, key=keys[2]),
            eqx.nn.Linear(32, 1, key=keys[3]),  # 1 output: u
        ]

    def __call__(self, xt):
        h = jax.nn.tanh(self.layers[0](xt))
        h = jax.nn.tanh(self.layers[1](h))
        h = jax.nn.tanh(self.layers[2](h))
        return self.layers[3](h)


# Initialize model
key = jax.random.PRNGKey(42)
pinn = PINN(key)


# --- 3. Domain Layer: The 1D Heat Equation PDE ---
# This function defines the PDE residual

def pde_residual(model, x_val, t_val):
    """Compute PDE residual for a single point (x, t)."""
    def u_fn(x, t):
        xt = jnp.array([x, t])
        return model(xt)[0]

    # Compute partial derivatives
    u_x = jax.grad(u_fn, argnums=0)(x_val, t_val)
    u_xx = jax.grad(jax.grad(u_fn, argnums=0), argnums=0)(x_val, t_val)
    u_t = jax.grad(u_fn, argnums=1)(x_val, t_val)

    # Calculate the PDE residual: u_t - alpha * u_xx
    pde_res = u_t - alpha * u_xx
    return pde_res


def pde_residual_batch(model, x_batch, t_batch):
    """Vectorized PDE residual for batch of points."""
    return jax.vmap(lambda x, t: pde_residual(model, x[0], t[0]))(x_batch, t_batch)


# --- 4. Physics-Informed Loss Functions ---

def loss_fn(model, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right):
    """Compute total loss."""
    # PDE Loss
    x_col = collocation_pts[:, 0:1]
    t_col = collocation_pts[:, 1:2]
    pde_res = pde_residual_batch(model, x_col, t_col)
    pde_loss = jnp.mean(pde_res ** 2)

    # IC Loss
    u_ic_pred = jax.vmap(lambda xt: model(xt)[0])(ic_pts).reshape(-1, 1)
    ic_loss = jnp.mean((u_ic_pred - u_ic_vals) ** 2)

    # BC Loss
    u_bc_pred_left = jax.vmap(lambda xt: model(xt)[0])(bc_pts_left).reshape(-1, 1)
    u_bc_pred_right = jax.vmap(lambda xt: model(xt)[0])(bc_pts_right).reshape(-1, 1)

    bc_loss = jnp.mean(u_bc_pred_left ** 2) + jnp.mean(u_bc_pred_right ** 2)

    # Total loss with weights
    loss = pde_loss + 10 * ic_loss + 10 * bc_loss

    return loss, (pde_loss, ic_loss, bc_loss)


@eqx.filter_jit
def train_step(model, opt_state, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right):
    """Single training step."""
    (loss, (pde_loss, ic_loss, bc_loss)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, pde_loss, ic_loss, bc_loss


# --- 5. Training and Results ---
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

epochs = 40000

print(f"Training started for {epochs} epochs...")

for epoch in range(epochs):
    pinn, opt_state, loss, pde_loss, ic_loss, bc_loss = train_step(
        pinn, opt_state, collocation_points, ic_points, u_ic, bc_points_left, bc_points_right
    )

    if (epoch + 1) % 2000 == 0:
        print(f"Epoch [{epoch + 1}/{epochs}], Total Loss: {loss:.6f}, "
              f"PDE Loss: {pde_loss:.6f}, IC Loss: {ic_loss:.6f}, "
              f"BC Loss: {bc_loss:.6f}")

print("Training finished!")

# --- Visualize Final Results ---
# Generate a grid of points for plotting
n_points = 100
x_plot = np.linspace(x_min, x_max, n_points)
t_plot = np.linspace(t_min, t_max, n_points)
X_plot, T_plot = np.meshgrid(x_plot, t_plot)
xt_plot_flat = jnp.concatenate([
    jnp.array(X_plot.flatten()).reshape(-1, 1),
    jnp.array(T_plot.flatten()).reshape(-1, 1)
], axis=1)

u_pred = jax.vmap(lambda xt: pinn(xt)[0])(xt_plot_flat)
u_pred = np.array(u_pred).reshape(n_points, n_points)

# 1. 3D Plot of PINN Solution
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_plot, T_plot, u_pred, cmap='viridis')
ax.set_title("PINN Solution for 1D Heat Equation")
ax.set_xlabel("x")
ax.set_ylabel("t")
ax.set_zlabel("u(x, t)")
plt.show()

# 2. Comparison Plot at a specific time slice
try:
    # Analytical solution for the heat equation
    def u_analytical(x, t, alpha, terms=20):
        sol = np.zeros_like(x)
        for n in range(1, terms + 1):
            term = (2 / np.pi) * (1 - (-1) ** n) / n
            sol += term * np.sin(n * np.pi * x) * np.exp(-n ** 2 * np.pi ** 2 * alpha * t)
        return sol


    t_slice = 0.5  # Choose a time to compare
    x_slice = np.linspace(x_min, x_max, n_points)
    u_analytical_slice = u_analytical(x_slice, t_slice, alpha)

    xt_slice = jnp.concatenate([
        jnp.array(x_slice).reshape(-1, 1),
        jnp.full((n_points, 1), t_slice)
    ], axis=1)
    u_pinn_slice = jax.vmap(lambda xt: pinn(xt)[0])(xt_slice)
    u_pinn_slice = np.array(u_pinn_slice)

    plt.figure(figsize=(10, 6))
    plt.plot(x_slice, u_analytical_slice, label='Analytical Solution', color='blue', linestyle='--')
    plt.plot(x_slice, u_pinn_slice, label='PINN Prediction', color='red')
    plt.title(f"Comparison at t = {t_slice}")
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    plt.legend()
    plt.grid(True)
    plt.show()

except Exception as e:
    print(f"Could not plot analytical solution due to error: {e}")
