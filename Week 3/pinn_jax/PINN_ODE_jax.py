"""
Vanilla PINN Example 4.1.1 (Damped Pendulum ODE) - JAX Implementation.

Example inspired by
https://benmoseley.blog/my-research/so-what-is-a-physics-informed-neural-network/
https://www.mathworks.com/discovery/physics-informed-neural-networks.html
"""

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import imageio
from PIL import Image
import io

# --- 1. Data: Sampled Collocation Points ---
# These are the time points where we will enforce the physics.

# Define physical parameters
g = 9.81  # Acceleration due to gravity
l = 1.0  # Pendulum length
beta = 0.5  # Damping coefficient

# Initial conditions
u0 = np.pi / 2  # Initial angular displacement (radians)
v0 = 0.0  # Initial angular velocity

# Time domain
t_min, t_max = 0.0, 10.0
num_collocation_points = 500
collocation_points = jnp.linspace(t_min, t_max, num_collocation_points).reshape(-1, 1)

# Initial condition points for loss calculation
t_ic = jnp.array([[0.0]])
u_ic = jnp.array([[u0]])
v_ic = jnp.array([[v0]])


# --- 2. Machine Learning Model: Neural Network ---
# This network will approximate the solution u(t).

class PINN(eqx.Module):
    layers: list

    def __init__(self, key):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, 32, key=keys[0]),
            eqx.nn.Linear(32, 32, key=keys[1]),
            eqx.nn.Linear(32, 1, key=keys[2]),
        ]

    def __call__(self, t):
        x = jax.nn.tanh(self.layers[0](t))
        x = jax.nn.tanh(self.layers[1](x))
        return self.layers[2](x)


# Initialize model
key = jax.random.PRNGKey(0)
pinn = PINN(key)


# --- 3. Domain Layer: The Damped Pendulum ODE ---
# This function calculates the residual of the ODE.
# It uses JAX's autograd to compute derivatives.

def physics_residual(model, t_val):
    """Compute ODE residual for a single time point."""
    def u_fn(t):
        return model(jnp.array([[t]]))[0, 0]

    u = u_fn(t_val)
    u_t = jax.grad(u_fn)(t_val)
    u_tt = jax.grad(jax.grad(u_fn))(t_val)

    # Define the ODE residual (should be close to zero)
    # d^2u/dt^2 + beta * du/dt + (g/l) * sin(u) = 0
    ode_residual = u_tt + beta * u_t + (g / l) * jnp.sin(u)

    return ode_residual


def physics_residual_batch(model, t_batch):
    """Vectorized physics residual for batch of time points."""
    return jax.vmap(lambda t: physics_residual(model, t[0]))(t_batch)


# --- 4. GIF visualisation functions ---

# Create a list to store plot images for the GIF
gif_frames = []
t_test = jnp.linspace(t_min, t_max, 500).reshape(-1, 1)
collocation_points_np = np.array(collocation_points)

# A reference numerical solution for comparison
try:
    from scipy.integrate import solve_ivp

    def damped_pendulum_ode(t, y, beta, g, l):
        theta, omega = y
        dtheta_dt = omega
        domega_dt = -beta * omega - (g / l) * np.sin(theta)
        return [dtheta_dt, domega_dt]

    sol = solve_ivp(
        damped_pendulum_ode,
        (t_min, t_max),
        [u0, v0],
        args=(beta, g, l),
        dense_output=True
    )
    t_ref = np.linspace(t_min, t_max, 500)
    u_ref = sol.sol(t_ref)[0]
except ImportError:
    sol, u_ref = None, None
    print("SciPy not found. Skipping numerical solution in GIF.")


# plotting function for gif
def save_frame(epoch, model):
    """Saves a plot frame for the GIF."""
    plt.figure(figsize=(10, 6))

    # Plotting the current PINN prediction
    u_pred = jax.vmap(lambda t: model(t)[0])(t_test)
    plt.plot(np.array(t_test), np.array(u_pred),
             label='PINN Prediction', color='red', linewidth=2)

    # Plotting the collocation points
    plt.scatter(collocation_points_np, np.zeros(collocation_points_np.shape),
                s=10, label='Collocation Points', color='orange', alpha=0.5)

    # Plotting the numerical solution if available
    if u_ref is not None:
        plt.plot(t_ref, u_ref, label='Numerical Solution (SciPy)',
                 linestyle='--', color='blue')

    plt.title(f"PINN Training Progression | Epoch {epoch}")
    plt.xlabel("Time (t)")
    plt.ylabel("Angular Displacement u(t)")
    plt.legend()
    plt.grid(True)

    # Save the figure to an in-memory buffer without closing it
    buf = io.BytesIO()
    plt.savefig(buf, format='png')

    # Reset the buffer position to the beginning and read the image
    buf.seek(0)
    img = Image.open(buf)

    # Append the image to the list and close the figure
    gif_frames.append(img)
    plt.close()


# --- 5. Physics-Informed Loss Functions, Training and Results ---
# We combine the initial condition loss and the physics loss.

def loss_fn(model, collocation_pts, t_ic_val, u_ic_val, v_ic_val):
    """Compute total loss."""
    # Calculate the physics loss
    ode_residuals = physics_residual_batch(model, collocation_pts)
    physics_loss = jnp.mean(ode_residuals ** 2)

    # Calculate the initial condition loss
    u_pred_ic = model(t_ic_val)[0, 0]

    def u_fn_ic(t):
        return model(jnp.array([[t]]))[0, 0]

    u_t_pred_ic = jax.grad(u_fn_ic)(t_ic_val[0, 0])

    ic_loss_pos = (u_pred_ic - u_ic_val[0, 0]) ** 2
    ic_loss_vel = (u_t_pred_ic - v_ic_val[0, 0]) ** 2
    ic_loss = ic_loss_pos + ic_loss_vel

    # Total loss is a combination of both
    total_loss = physics_loss + ic_loss

    return total_loss, (physics_loss, ic_loss)


@eqx.filter_jit
def train_step(model, opt_state, collocation_pts, t_ic_val, u_ic_val, v_ic_val):
    """Single training step."""
    (loss, (phys_loss, ic_loss_val)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, collocation_pts, t_ic_val, u_ic_val, v_ic_val
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, phys_loss, ic_loss_val


# Setup optimizer
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

epochs = 30000

print(f"Training started for {epochs} epochs...")

for epoch in range(epochs):
    pinn, opt_state, total_loss, physics_loss, ic_loss = train_step(
        pinn, opt_state, collocation_points, t_ic, u_ic, v_ic
    )

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {total_loss:.6f}, '
              f'Physics Loss: {physics_loss:.6f}, IC Loss: {ic_loss:.6f}')
        # Save a frame for the GIF every 100 epochs
        save_frame(epoch + 1, pinn)

print("Training finished!")

# --- Visualize Final Results and Generate GIF ---
# Final plot with collocation points
u_pred_final = jax.vmap(lambda t: pinn(t)[0])(t_test)

plt.figure(figsize=(10, 6))
plt.plot(np.array(t_test), np.array(u_pred_final), label='PINN Prediction', color='red', linewidth=2)

u_collocation = jax.vmap(lambda t: pinn(t)[0])(collocation_points)
plt.scatter(collocation_points_np, np.array(u_collocation), s=10, label='Collocation Points',
            color='red', alpha=0.5)

if u_ref is not None:
    plt.plot(t_ref, u_ref, label='Numerical Solution (SciPy)', linestyle='--', color='blue')

plt.title("Damped Pendulum: Final PINN Solution")
plt.xlabel("Time (t)")
plt.ylabel("Angular Displacement u(t)")
plt.legend()
plt.grid(True)
plt.show()

# Generate the GIF from the collected frames
if gif_frames:
    imageio.mimsave('pinn_training_progression.gif', gif_frames, fps=20)
    print("GIF saved as pinn_training_progression.gif")
