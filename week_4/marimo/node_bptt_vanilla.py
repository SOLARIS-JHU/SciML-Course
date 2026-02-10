# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(app_title="Neural ODE - Vanilla")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural Ordinary Differential Equations (NODEs)
    ## Learning Dynamical Systems with Black-Box Neural Networks

    **Neural ODEs** learn continuous-time dynamics directly from trajectory data:
    - Replace known physics $\frac{dx}{dt} = f(x; \theta)$ with neural network $\frac{dx}{dt} = \text{NN}_\phi(x)$
    - Train via backpropagation through time (BPTT) on observed trajectories
    - No physics assumptions required—purely data-driven

    | Traditional ODE Solvers | Neural ODEs |
    |------------------------|-------------|
    | Require known equations | **Learn** dynamics from data |
    | Fixed model structure | **Flexible** neural network |
    | Domain expertise needed | **Data-driven** discovery |
    | No parameter learning | **End-to-end** optimization |

    **This Version:** Vanilla implementation with basic RK4 integration and MSE loss.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    import numpy as np
    return mo, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The Physical System

    **Damped Pendulum Dynamics:**

    $$\frac{d\theta}{dt} = \omega, \quad \frac{d\omega}{dt} = -\beta \omega - \frac{g}{\ell}\sin(\theta)$$

    **State Vector:** $x = [\theta, \omega]$ (angle, angular velocity)

    **Goal:** Learn $\frac{dx}{dt} = f_{\text{NN}}(x)$ without knowing the analytical form.
    """)
    return


@app.cell
def _(mo, np, torch):
    @mo.cache
    def generate_training_data(M, T, N, g, beta, ell, noise_std, seed, device_type):
        """Generate synthetic pendulum trajectories with noise"""
        device = torch.device(device_type)
        torch.manual_seed(seed)

        # Time grid
        t_grid = torch.linspace(0.0, T, N, device=device)

        # Sample random initial conditions
        u0 = 0.5 * (2*torch.rand(M, device=device)-1)
        v0 = 0.5 * (2*torch.rand(M, device=device)-1)
        x0_batch = torch.stack([u0, v0], dim=-1)

        # Pendulum dynamics
        def pendulum_rhs(x, beta, ell, g):
            u, v = x[..., 0], x[..., 1]
            du = v
            dv = -beta * v - (g / ell) * torch.sin(u)
            return torch.stack([du, dv], dim=-1)

        # RK4 integrator
        def rk4_integrate(f, x0, t, *f_args):
            M = x0.shape[0]
            N = t.shape[0]
            x = torch.zeros(N, M, 2, device=x0.device, dtype=x0.dtype)
            x = x.clone()
            x = torch.index_copy(x, 0, torch.tensor(0, device=x0.device), x0.unsqueeze(0))

            for k in range(N-1):
                h = t[k+1] - t[k]
                xk = x[k]
                k1 = f(xk, *f_args)
                k2 = f(xk + 0.5*h*k1, *f_args)
                k3 = f(xk + 0.5*h*k2, *f_args)
                k4 = f(xk + h*k3, *f_args)
                x_next = xk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
                x = torch.index_copy(x, 0, torch.tensor(k+1, device=x0.device), x_next.unsqueeze(0))
            return x

        # Generate true trajectories
        with torch.no_grad():
            x_true = rk4_integrate(lambda x, b, l: pendulum_rhs(x, b, l, g),
                                   x0_batch, t_grid, beta, ell)

        # Add noise
        noise = noise_std * torch.randn_like(x_true)
        x_obs = (x_true + noise).detach()

        return {
            't_grid': t_grid.cpu().numpy(),
            'x0_batch': x0_batch,
            'x_true': x_true,
            'x_obs': x_obs
        }
    return (generate_training_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. Neural ODE Architecture

    **Black-box dynamics learner:**
    - Input: State $x \in \mathbb{R}^2$
    - Output: Derivative $\dot{x} \in \mathbb{R}^2$
    - Architecture: MLP with 2 hidden layers
    """)
    return


@app.cell
def _(nn, torch):
    def rk4_integrate_simple(f, x0, t):
        """RK4 integrator for neural network dynamics"""
        M = x0.shape[0]
        N = t.shape[0]
        device = x0.device
        x = torch.zeros(N, M, 2, device=device, dtype=x0.dtype)
        x = x.clone()
        x = torch.index_copy(x, 0, torch.tensor(0, device=device), x0.unsqueeze(0))

        for k in range(N-1):
            h = t[k+1] - t[k]
            xk = x[k]
            k1 = f(xk)
            k2 = f(xk + 0.5*h*k1)
            k3 = f(xk + 0.5*h*k2)
            k4 = f(xk + h*k3)
            x_next = xk + (h/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            x = torch.index_copy(x, 0, torch.tensor(k+1, device=device), x_next.unsqueeze(0))
        return x

    class NeuralODE(nn.Module):
        def __init__(self, hidden_dim=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

        def forward(self, x0, t):
            return rk4_integrate_simple(lambda x: self.net(x), x0, t)
    return NeuralODE, rk4_integrate_simple


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. Training Configuration

    Adjust hyperparameters below. Training uses backpropagation through the RK4 integrator.
    """)
    return


@app.cell
def _(mo):
    # Data generation parameters
    M_slider = mo.ui.slider(8, 64, value=32, step=8, label="Number of trajectories (M)")
    T_slider = mo.ui.slider(2.0, 10.0, value=5.0, step=1.0, label="Time horizon (T)")
    N_slider = mo.ui.slider(50, 500, value=200, step=50, label="Time steps (N)")

    # Physical parameters (for data generation only)
    g_slider = mo.ui.slider(5.0, 15.0, value=9.81, step=0.1, label="Gravity g (m/s²)")
    beta_slider = mo.ui.slider(0.0, 1.0, value=0.25, step=0.05, label="Damping β")
    ell_slider = mo.ui.slider(0.5, 2.0, value=0.9, step=0.1, label="Length ℓ (m)")
    noise_slider = mo.ui.slider(0.0, 0.05, value=0.01, step=0.005, label="Observation noise")

    # Training parameters
    hidden_dim_slider = mo.ui.slider(32, 256, value=64, step=32, label="Hidden dimension")
    epochs_dropdown = mo.ui.dropdown({"100": 100, "250": 250, "500": 500, "1000": 1000},
                                      value="500", label="Epochs")
    lr_dropdown = mo.ui.dropdown({"1e-3": 1e-3, "5e-3": 5e-3, "1e-2": 1e-2, "5e-2": 5e-2},
                                  value="5e-2", label="Learning rate")

    # Random seed
    seed_slider = mo.ui.slider(0, 9999, value=0, step=1, label="Random seed")
    return (
        M_slider,
        N_slider,
        T_slider,
        beta_slider,
        ell_slider,
        epochs_dropdown,
        g_slider,
        hidden_dim_slider,
        lr_dropdown,
        noise_slider,
        seed_slider,
    )


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    beta_slider,
    ell_slider,
    epochs_dropdown,
    g_slider,
    hidden_dim_slider,
    lr_dropdown,
    mo,
    noise_slider,
    seed_slider,
    torch,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    control_panel = mo.vstack([
        mo.md("#### Data Generation"),
        M_slider, T_slider, N_slider,
        mo.md("#### True Physical Parameters"),
        g_slider, beta_slider, ell_slider, noise_slider,
        mo.md("#### Neural Network"),
        hidden_dim_slider,
        mo.md("#### Training"),
        epochs_dropdown, lr_dropdown, seed_slider,
        mo.md("---"),
        mo.md(f"**Device:** `{device}`")
    ])

    train_button = mo.ui.run_button(label="▶ Train Neural ODE")

    mo.vstack([train_button, control_panel])
    return control_panel, device, train_button


@app.cell
def _(
    M_slider,
    N_slider,
    NeuralODE,
    T_slider,
    beta_slider,
    device,
    ell_slider,
    epochs_dropdown,
    g_slider,
    generate_training_data,
    hidden_dim_slider,
    lr_dropdown,
    mo,
    nn,
    noise_slider,
    np,
    seed_slider,
    torch,
    train_button,
):
    mo.stop(not train_button.value, mo.md("_Click **▶ Train Neural ODE** to begin_"))

    # Generate data
    data = generate_training_data(
        M=M_slider.value, T=T_slider.value, N=N_slider.value,
        g=g_slider.value, beta=beta_slider.value, ell=ell_slider.value,
        noise_std=noise_slider.value, seed=seed_slider.value,
        device_type=device.type
    )

    t_grid_tensor = torch.tensor(data['t_grid'], device=device, dtype=torch.float32)
    x0_batch = data['x0_batch']
    x_obs = data['x_obs']

    # Initialize model
    model = NeuralODE(hidden_dim=hidden_dim_slider.value).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dropdown.value)
    mse = nn.MSELoss()

    # Training loop
    losses = []
    print(f"Training Neural ODE for {epochs_dropdown.value} epochs...")

    for epoch in range(epochs_dropdown.value):
        optimizer.zero_grad()
        x_sim = model(x0_batch, t_grid_tensor)
        loss = mse(x_sim, x_obs)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if epoch % 100 == 0:
            print(f"[{epoch:04d}] loss={loss.item():.6f}")

    print("Training complete!")

    # Final evaluation
    with torch.no_grad():
        x_fit = model(x0_batch, t_grid_tensor)

    results = {
        'model': model,
        'losses': np.array(losses),
        't_grid': data['t_grid'],
        'x_true': data['x_true'].cpu().numpy(),
        'x_obs': data['x_obs'].cpu().numpy(),
        'x_fit': x_fit.cpu().numpy(),
        'M': M_slider.value
    }
    return (
        data,
        epoch,
        loss,
        losses,
        model,
        mse,
        optimizer,
        results,
        t_grid_tensor,
        x0_batch,
        x_fit,
        x_obs,
        x_sim,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 4. Results Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.1 Training Loss Evolution
    """)
    return


@app.cell(hide_code=True)
def _(plt, results):
    fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(results['losses'], 'b-', linewidth=2)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('MSE Loss (log scale)')
    ax1.set_title('Training Loss Evolution')
    ax1.grid(True, alpha=0.3)

    start_idx = int(0.5 * len(results['losses']))
    ax2.plot(range(start_idx, len(results['losses'])), results['losses'][start_idx:], 'b-', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Loss (Last 50%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_loss
    return ax1, ax2, fig_loss, start_idx


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.2 Trajectory Comparison
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, results, torch):
    # Select 3 random trajectories
    num_traj = min(3, results['M'])
    indices = torch.randperm(results['M'])[:num_traj].numpy()

    fig_traj, axes = plt.subplots(2, num_traj, figsize=(5*num_traj, 8), sharex=True)

    # Ensure axes is 2D
    if num_traj == 1:
        axes = axes.reshape(2, 1)

    for i, idx in enumerate(indices):
        # Angle (u)
        axes[0, i].plot(results['t_grid'], results['x_true'][:, idx, 0], 'b-', linewidth=2, label='True', alpha=0.8)
        axes[0, i].scatter(results['t_grid'], results['x_obs'][:, idx, 0], s=10, c='gray', alpha=0.4, label='Observed')
        axes[0, i].plot(results['t_grid'], results['x_fit'][:, idx, 0], 'r--', linewidth=2, label='Neural ODE')
        axes[0, i].set_ylabel('Angle θ (rad)', fontsize=11)
        axes[0, i].set_title(f'Trajectory {idx+1}: Angle', fontsize=12)
        axes[0, i].legend(loc='best', fontsize=9)
        axes[0, i].grid(True, alpha=0.3)

        # Angular velocity (v)
        axes[1, i].plot(results['t_grid'], results['x_true'][:, idx, 1], 'b-', linewidth=2, label='True', alpha=0.8)
        axes[1, i].scatter(results['t_grid'], results['x_obs'][:, idx, 1], s=10, c='gray', alpha=0.4, label='Observed')
        axes[1, i].plot(results['t_grid'], results['x_fit'][:, idx, 1], 'r--', linewidth=2, label='Neural ODE')
        axes[1, i].set_xlabel('Time (s)', fontsize=11)
        axes[1, i].set_ylabel('Velocity ω (rad/s)', fontsize=11)
        axes[1, i].set_title(f'Trajectory {idx+1}: Velocity', fontsize=12)
        axes[1, i].legend(loc='best', fontsize=9)
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_traj
    return axes, fig_traj, i, idx, indices, num_traj


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.3 Phase Portrait
    """)
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    # Phase portraits for all trajectories
    fig_phase, ax_phase = plt.subplots(figsize=(10, 10))

    for idx in range(min(10, results['M'])):
        ax_phase.plot(results['x_true'][:, idx, 0], results['x_true'][:, idx, 1],
                     'b-', linewidth=1, alpha=0.3, label='True' if idx == 0 else '')
        ax_phase.plot(results['x_fit'][:, idx, 0], results['x_fit'][:, idx, 1],
                     'r--', linewidth=1, alpha=0.5, label='Neural ODE' if idx == 0 else '')

    ax_phase.set_xlabel('Angle θ (rad)', fontsize=12)
    ax_phase.set_ylabel('Angular Velocity ω (rad/s)', fontsize=12)
    ax_phase.set_title('Phase Portrait: True vs Neural ODE', fontsize=14, fontweight='bold')
    ax_phase.grid(True, alpha=0.3)
    ax_phase.legend()
    ax_phase.axhline(0, color='k', linestyle='-', alpha=0.2)
    ax_phase.axvline(0, color='k', linestyle='-', alpha=0.2)

    plt.tight_layout()
    fig_phase
    return ax_phase, fig_phase, idx


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 4.4 Error Metrics
    """)
    return


@app.cell(hide_code=True)
def _(mo, np, results):
    mae_angle = np.mean(np.abs(results['x_fit'][:, :, 0] - results['x_true'][:, :, 0]))
    mae_velocity = np.mean(np.abs(results['x_fit'][:, :, 1] - results['x_true'][:, :, 1]))

    rmse_angle = np.sqrt(np.mean((results['x_fit'][:, :, 0] - results['x_true'][:, :, 0])**2))
    rmse_velocity = np.sqrt(np.mean((results['x_fit'][:, :, 1] - results['x_true'][:, :, 1])**2))

    max_error_angle = np.max(np.abs(results['x_fit'][:, :, 0] - results['x_true'][:, :, 0]))
    max_error_velocity = np.max(np.abs(results['x_fit'][:, :, 1] - results['x_true'][:, :, 1]))

    metrics_table = mo.ui.table([
        {"Metric": "Angle MAE", "Value": f"{mae_angle:.4e}"},
        {"Metric": "Velocity MAE", "Value": f"{mae_velocity:.4e}"},
        {"Metric": "Angle RMSE", "Value": f"{rmse_angle:.4e}"},
        {"Metric": "Velocity RMSE", "Value": f"{rmse_velocity:.4e}"},
        {"Metric": "Angle Max Error", "Value": f"{max_error_angle:.4e}"},
        {"Metric": "Velocity Max Error", "Value": f"{max_error_velocity:.4e}"},
        {"Metric": "Final Loss", "Value": f"{results['losses'][-1]:.4e}"}
    ], selection=None)

    metrics_table
    return (
        mae_angle,
        mae_velocity,
        max_error_angle,
        max_error_velocity,
        metrics_table,
        rmse_angle,
        rmse_velocity,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    **Neural ODE Vanilla Implementation:**
    - **Strengths:** Simple architecture, pure data-driven learning
    - **Limitations:** May converge to trivial solutions, sensitive to initialization
    - **Use Cases:** Learning unknown dynamics from trajectory data

    **Key Challenges:**
    - Backpropagation through ODE solver (memory intensive)
    - Trivial solutions (zero derivatives)
    - Hyperparameter sensitivity

    **Next Steps:** See tuned version for improvements (normalization, regularization, derivative penalties)
    """)
    return


if __name__ == "__main__":
    app.run()
