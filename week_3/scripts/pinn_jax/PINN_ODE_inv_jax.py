"""
Inverse PINN for the Damped Pendulum ODE - JAX Implementation:
Estimate the damping coefficient beta from measurement data.

Changes vs. vanilla:
- beta_hat is a learnable parameter.
- Adds data misfit loss using (t_meas, u_meas).
- Keeps physics (collocation) + IC losses.
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

# -------------------------------
# 0) Config & Utilities
# -------------------------------

# Physical constants (g, l) are assumed known
g = 9.81
l = 1.0

# True beta for synthetic demo (ignored if you provide your own data)
beta_true = 0.5

# Initial conditions
u0 = np.pi / 2
v0 = 0.0

# Time domain
t_min, t_max = 0.0, 10.0

# Collocation points (physics)
num_collocation_points = 500
collocation_points = jnp.linspace(t_min, t_max, num_collocation_points).reshape(-1, 1)

# IC tensors
t_ic = jnp.array([[0.0]])
u_ic = jnp.array([[u0]])
v_ic = jnp.array([[v0]])

# Loss weights
w_phys, w_ic, w_data = 1.0, 1.0, 5.0  # increase data weight for inverse tasks (tune as needed)

# Measurement data options
use_synthetic_measurements = True     # set False if you will provide your own (t_meas_np, u_meas_np)
num_meas = 50
meas_noise_std = 0.02                 # std dev of Gaussian noise for synthetic demo

# Training
epochs = 20000
lr = 1e-3
print_every = 200

# GIF settings
make_gif = True
gif_every = 200
gif_frames = []
t_test = jnp.linspace(t_min, t_max, 500).reshape(-1, 1)

# --------------------------------
# 1) Reference solver (for demo/plot)
# --------------------------------
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
        args=(beta_true, g, l),
        dense_output=True
    )
    t_ref = np.linspace(t_min, t_max, 500)
    u_ref = sol.sol(t_ref)[0]
except ImportError:
    sol, u_ref = None, None
    print("SciPy not found. Skipping numerical reference.")

# --------------------------------
# 2) Measurement data (provide or synthesize)
# --------------------------------
if use_synthetic_measurements:
    rng = np.random.default_rng(0)
    t_meas_np = np.sort(rng.uniform(t_min, t_max, size=num_meas))
    if sol is not None:
        u_clean = sol.sol(t_meas_np)[0]
    else:
        # fallback: small subset from reference-less case (not ideal)
        t_meas_np = np.linspace(t_min, t_max, num_meas)
        u_clean = np.interp(t_meas_np, t_ref, u_ref) if u_ref is not None else np.zeros_like(t_meas_np)
    u_meas_np = u_clean + rng.normal(0.0, meas_noise_std, size=num_meas)
else:
    # If you have real measurements, set these:
    # t_meas_np = np.array([...], dtype=float)
    # u_meas_np = np.array([...], dtype=float)
    raise ValueError("Set use_synthetic_measurements=True or provide your own measurements.")

t_meas = jnp.array(t_meas_np).reshape(-1, 1)
u_meas = jnp.array(u_meas_np).reshape(-1, 1)

# --------------------------------
# 3) PINN model
# --------------------------------
class InversePINN(eqx.Module):
    layers: list
    beta_hat: jax.Array  # Learnable parameter

    def __init__(self, key, width=32):
        keys = jax.random.split(key, 3)
        self.layers = [
            eqx.nn.Linear(1, width, key=keys[0]),
            eqx.nn.Linear(width, width, key=keys[1]),
            eqx.nn.Linear(width, 1, key=keys[2]),
        ]
        # Learnable beta (initialized reasonably)
        self.beta_hat = jnp.array(0.1)

    def __call__(self, t):
        x = jax.nn.tanh(self.layers[0](t))
        x = jax.nn.tanh(self.layers[1](x))
        return self.layers[2](x)


key = jax.random.PRNGKey(0)
pinn = InversePINN(key, width=32)

# --------------------------------
# 4) Physics residual
# --------------------------------
def physics_residual(model, t_val):
    """Compute ODE residual for a single time point."""
    def u_fn(t):
        return model(jnp.array([[t]]))[0, 0]

    u = u_fn(t_val)
    u_t = jax.grad(u_fn)(t_val)
    u_tt = jax.grad(jax.grad(u_fn))(t_val)

    beta = model.beta_hat
    ode_res = u_tt + beta * u_t + (g / l) * jnp.sin(u)
    return ode_res


def physics_residual_batch(model, t_batch):
    """Vectorized physics residual for batch of time points."""
    return jax.vmap(lambda t: physics_residual(model, t[0]))(t_batch)


# --------------------------------
# 5) Training
# --------------------------------

def loss_fn(model, collocation_pts, t_ic_val, u_ic_val, v_ic_val, t_meas_val, u_meas_val):
    """Compute total loss."""
    # Physics loss on collocation points
    ode_residuals = physics_residual_batch(model, collocation_pts)
    physics_loss = jnp.mean(ode_residuals ** 2)

    # IC loss
    u_pred_ic = model(t_ic_val)[0, 0]

    def u_fn_ic(t):
        return model(jnp.array([[t]]))[0, 0]

    u_t_pred_ic = jax.grad(u_fn_ic)(t_ic_val[0, 0])

    ic_loss = (u_pred_ic - u_ic_val[0, 0]) ** 2 + (u_t_pred_ic - v_ic_val[0, 0]) ** 2

    # Data misfit loss
    u_pred_meas = jax.vmap(lambda t: model(t)[0])(t_meas_val).reshape(-1, 1)
    data_loss = jnp.mean((u_pred_meas - u_meas_val) ** 2)

    # Total loss
    total_loss = w_phys * physics_loss + w_ic * ic_loss + w_data * data_loss

    return total_loss, (physics_loss, ic_loss, data_loss)


@eqx.filter_jit
def train_step(model, opt_state, collocation_pts, t_ic_val, u_ic_val, v_ic_val, t_meas_val, u_meas_val):
    """Single training step."""
    (loss, (phys_loss, ic_loss, data_loss)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, collocation_pts, t_ic_val, u_ic_val, v_ic_val, t_meas_val, u_meas_val
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, phys_loss, ic_loss, data_loss


def save_frame(epoch, model):
    """Saves a plot frame for the GIF."""
    plt.figure(figsize=(10, 6))

    # Current PINN prediction
    u_pred = jax.vmap(lambda t: model(t)[0])(t_test)
    plt.plot(np.array(t_test), np.array(u_pred), label='PINN Prediction', linewidth=2)

    # Measurements
    plt.scatter(np.array(t_meas), np.array(u_meas), s=20, label='Measurements', alpha=0.7)

    # Reference solution (if available)
    if u_ref is not None:
        plt.plot(t_ref, u_ref, '--', label='Numerical (SciPy)')

    plt.title(f"Inverse PINN | Epoch {epoch} | beta_hat={float(model.beta_hat):.4f}")
    plt.xlabel("Time t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.grid(True)

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    gif_frames.append(Image.open(buf))
    plt.close()


optimizer = optax.adam(lr)
opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

print(f"Training started for {epochs} epochs...")
for epoch in range(1, epochs + 1):
    pinn, opt_state, total_loss, physics_loss, ic_loss, data_loss = train_step(
        pinn, opt_state, collocation_points, t_ic, u_ic, v_ic, t_meas, u_meas
    )

    if epoch % print_every == 0:
        print(f"Epoch [{epoch}/{epochs}] "
              f"Loss={total_loss:.6e} | "
              f"Phys={physics_loss:.3e} | IC={ic_loss:.3e} | Data={data_loss:.3e} | "
              f"beta_hat={float(pinn.beta_hat):.5f}")
        if make_gif and (epoch % gif_every == 0):
            save_frame(epoch, pinn)

print("Training finished!")
print(f"Estimated beta: {float(pinn.beta_hat):.6f} (true {beta_true:.6f})")

# --------------------------------
# 6) Final plots + GIF
# --------------------------------
u_pred_final = jax.vmap(lambda t: pinn(t)[0])(t_test)

plt.figure(figsize=(10, 6))
plt.plot(np.array(t_test), np.array(u_pred_final), label='PINN Prediction', linewidth=2)
plt.scatter(np.array(t_meas), np.array(u_meas), s=25, label='Measurements', alpha=0.8)
if u_ref is not None:
    plt.plot(t_ref, u_ref, '--', label='Numerical (SciPy)')
plt.title(f"Inverse PINN Result (beta_hat={float(pinn.beta_hat):.4f})")
plt.xlabel("Time (t)")
plt.ylabel("Angular Displacement u(t)")
plt.legend()
plt.grid(True)
plt.show()

if make_gif and gif_frames:
    imageio.mimsave('inverse_pinn_training.gif', gif_frames, fps=10)
    print("GIF saved as inverse_pinn_training.gif")
