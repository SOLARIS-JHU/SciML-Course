# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(app_title="Neural ODE - Tuned")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Neural ODEs - Tuned Implementation
    ## Advanced Training Strategies for Robust Learning

    **This version includes critical improvements** to overcome common Neural ODE pitfalls:

    | Problem | Vanilla | Tuned Solution |
    |---------|---------|----------------|
    | **Trivial solutions** | Network learns $\dot{x} \approx 0$ | ✅ Derivative norm penalty |
    | **Scale mismatch** | Angle/velocity different scales | ✅ Data normalization |
    | **Vanishing gradients** | Random initialization | ✅ He/Kaiming initialization |
    | **Overfitting** | No regularization | ✅ Weight decay (L2) |
    | **Poor convergence** | Fixed learning rate | ✅ LR scheduler |
    | **Gradient explosion** | Unbounded gradients | ✅ Gradient clipping |

    **Key Insight:** Neural ODEs can easily converge to $\frac{dx}{dt} = 0$, which trivially minimizes trajectory error but learns nothing useful. We combat this with explicit derivative penalties.
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

    **Training Objective:** Learn $\frac{dx}{dt} = f_{\text{NN}}(x)$ with **non-trivial** dynamics.
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

        # ✅ IMPROVEMENT: Normalize data to zero mean and unit variance
        x_obs_mean = x_obs.mean(dim=[0, 1], keepdim=True)
        x_obs_std = x_obs.std(dim=[0, 1], keepdim=True) + 1e-6
        x_obs_normalized = (x_obs - x_obs_mean) / x_obs_std
        x0_batch_normalized = (x0_batch - x_obs_mean[0]) / x_obs_std[0]

        return {
            't_grid': t_grid.cpu().numpy(),
            'x0_batch': x0_batch_normalized,
            'x_true': x_true,
            'x_obs': x_obs,
            'x_obs_normalized': x_obs_normalized,
            'x_obs_mean': x_obs_mean,
            'x_obs_std': x_obs_std
        }
    return (generate_training_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 2. Enhanced Neural ODE Architecture

    **Improvements:**
    1. ✅ Larger hidden dimension (128 vs 64)
    2. ✅ He/Kaiming initialization for ReLU activations
    3. ✅ Zero-initialized biases
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

    class TunedNeuralODE(nn.Module):
        def __init__(self, hidden_dim=128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2)
            )

            # ✅ IMPROVEMENT: He/Kaiming initialization
            for module in self.net:
                if isinstance(module, nn.Linear):
                    nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

        def forward(self, x0, t):
            return rk4_integrate_simple(lambda x: self.net(x), x0, t)
    return TunedNeuralODE, rk4_integrate_simple


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 3. Training Configuration

    **Enhanced training features:**
    - Weight decay (L2 regularization)
    - Learning rate scheduler
    - Gradient clipping
    - Derivative norm penalty

    $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MSE}} + \lambda \frac{1}{\|\dot{x}\|_2 + \epsilon}$$

    The penalty term discourages trivial solutions where $\dot{x} \to 0$.
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
    hidden_dim_slider = mo.ui.slider(32, 256, value=128, step=32, label="Hidden dimension")
    epochs_dropdown = mo.ui.dropdown({"250": 250, "500": 500, "1000": 1000, "2000": 2000},
                                      value="1000", label="Epochs")
    lr_dropdown = mo.ui.dropdown({"1e-3": 1e-3, "5e-3": 5e-3, "1e-2": 1e-2},
                                  value="1e-2", label="Initial learning rate")

    # Regularization parameters
    weight_decay_slider = mo.ui.slider(0.0, 1e-3, value=1e-4, step=1e-5, label="Weight decay", show_value=True)
    deriv_penalty_slider = mo.ui.slider(0.0, 0.1, value=0.01, step=0.005, label="Derivative penalty λ")
    grad_clip_slider = mo.ui.slider(0.1, 5.0, value=1.0, step=0.1, label="Gradient clip norm")

    # Random seed
    seed_slider = mo.ui.slider(0, 9999, value=0, step=1, label="Random seed")
    return (
        M_slider,
        N_slider,
        T_slider,
        beta_slider,
        deriv_penalty_slider,
        ell_slider,
        epochs_dropdown,
        g_slider,
        grad_clip_slider,
        hidden_dim_slider,
        lr_dropdown,
        noise_slider,
        seed_slider,
        weight_decay_slider,
    )


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    beta_slider,
    deriv_penalty_slider,
    ell_slider,
    epochs_dropdown,
    g_slider,
    grad_clip_slider,
    hidden_dim_slider,
    lr_dropdown,
    mo,
    noise_slider,
    seed_slider,
    torch,
    weight_decay_slider,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    control_panel = mo.vstack([
        mo.md("#### Data Generation"),
        M_slider, T_slider, N_slider,
        mo.md("#### True Physical Parameters"),
        g_slider, beta_slider, ell_slider, noise_slider,
        mo.md("#### Neural Network"),
        hidden_dim_slider,
        mo.md("#### Training & Regularization"),
        epochs_dropdown, lr_dropdown, weight_decay_slider,
        mo.md("#### Advanced Settings"),
        deriv_penalty_slider, grad_clip_slider, seed_slider,
        mo.md("---"),
        mo.md(f"**Device:** `{device}`")
    ])

    train_button = mo.ui.run_button(label="▶ Train Tuned Neural ODE")

    mo.vstack([train_button, control_panel])
    return control_panel, device, train_button


@app.cell
def _(
    M_slider,
    N_slider,
    T_slider,
    TunedNeuralODE,
    beta_slider,
    deriv_penalty_slider,
    device,
    ell_slider,
    epochs_dropdown,
    g_slider,
    generate_training_data,
    grad_clip_slider,
    hidden_dim_slider,
    lr_dropdown,
    mo,
    nn,
    noise_slider,
    np,
    seed_slider,
    torch,
    train_button,
    weight_decay_slider,
):
    mo.stop(not train_button.value, mo.md("_Click **▶ Train Tuned Neural ODE** to begin_"))

    # Generate data
    data = generate_training_data(
        M=M_slider.value, T=T_slider.value, N=N_slider.value,
        g=g_slider.value, beta=beta_slider.value, ell=ell_slider.value,
        noise_std=noise_slider.value, seed=seed_slider.value,
        device_type=device.type
    )

    t_grid_tensor = torch.tensor(data['t_grid'], device=device, dtype=torch.float32)
    x0_batch = data['x0_batch']
    x_obs_normalized = data['x_obs_normalized']
    x_obs = data['x_obs']
    x_obs_mean = data['x_obs_mean']
    x_obs_std = data['x_obs_std']

    # Initialize model
    model = TunedNeuralODE(hidden_dim=hidden_dim_slider.value).to(device)

    # ✅ IMPROVEMENT: Adam with weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dropdown.value,
                                  weight_decay=weight_decay_slider.value)

    # ✅ IMPROVEMENT: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                             factor=0.5, patience=100, verbose=False)

    mse = nn.MSELoss()

    # Helper function to compute derivative norm
    def compute_derivative_norm(model, x_sim):
        """Compute mean L2 norm of derivatives"""
        with torch.enable_grad():
            x_sim_clone = x_sim.clone().requires_grad_(True)
            dx_dt = model.net(x_sim_clone)
            norm = torch.mean(torch.norm(dx_dt, dim=-1))
        return norm

    # Training loop
    losses = []
    mse_losses = []
    deriv_norms = []
    lrs = []

    print(f"Training Tuned Neural ODE for {epochs_dropdown.value} epochs...")
    print(f"Weight decay: {weight_decay_slider.value:.2e}, Deriv penalty: {deriv_penalty_slider.value:.3f}, Grad clip: {grad_clip_slider.value:.2f}")

    for epoch in range(epochs_dropdown.value):
        optimizer.zero_grad()

        # Forward pass (in normalized space)
        x_sim = model(x0_batch, t_grid_tensor)

        # Rescale to original space for loss computation
        x_sim_rescaled = x_sim * x_obs_std + x_obs_mean

        # MSE loss
        mse_loss = mse(x_sim_rescaled, x_obs)

        # ✅ IMPROVEMENT: Derivative norm penalty to avoid trivial solutions
        deriv_norm = compute_derivative_norm(model, x_sim)
        penalty = deriv_penalty_slider.value * (1.0 / (deriv_norm + 1e-6))

        # Total loss
        total_loss = mse_loss + penalty

        total_loss.backward()

        # ✅ IMPROVEMENT: Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_slider.value)

        optimizer.step()

        # ✅ IMPROVEMENT: Step scheduler
        scheduler.step(mse_loss)

        # Logging
        losses.append(total_loss.item())
        mse_losses.append(mse_loss.item())
        deriv_norms.append(deriv_norm.item())
        lrs.append(optimizer.param_groups[0]['lr'])

        if epoch % 100 == 0:
            print(f"[{epoch:04d}] mse={mse_loss.item():.6f}  deriv_norm={deriv_norm.item():.6f}  lr={optimizer.param_groups[0]['lr']:.6f}")

    print("Training complete!")

    # Final evaluation
    with torch.no_grad():
        x_fit = model(x0_batch, t_grid_tensor)
        x_fit_rescaled = x_fit * x_obs_std + x_obs_mean

    results = {
        'model': model,
        'losses': np.array(losses),
        'mse_losses': np.array(mse_losses),
        'deriv_norms': np.array(deriv_norms),
        'lrs': np.array(lrs),
        't_grid': data['t_grid'],
        'x_true': data['x_true'].cpu().numpy(),
        'x_obs': data['x_obs'].cpu().numpy(),
        'x_fit': x_fit_rescaled.cpu().numpy(),
        'M': M_slider.value
    }
    return (
        compute_derivative_norm,
        data,
        deriv_norm,
        deriv_norms,
        epoch,
        losses,
        lrs,
        model,
        mse,
        mse_loss,
        mse_losses,
        optimizer,
        penalty,
        results,
        scheduler,
        t_grid_tensor,
        total_loss,
        x0_batch,
        x_fit,
        x_fit_rescaled,
        x_obs,
        x_obs_mean,
        x_obs_normalized,
        x_obs_std,
        x_sim,
        x_sim_clone,
        x_sim_rescaled,
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
    ### 4.1 Training Dynamics
    """)
    return


@app.cell(hide_code=True)
def _(plt, results):
    fig_train, axes_train = plt.subplots(2, 2, figsize=(14, 10))

    # MSE Loss (log scale)
    axes_train[0, 0].plot(results['mse_losses'], 'b-', linewidth=2)
    axes_train[0, 0].set_yscale('log')
    axes_train[0, 0].set_xlabel('Epoch')
    axes_train[0, 0].set_ylabel('MSE Loss (log)')
    axes_train[0, 0].set_title('MSE Loss Evolution')
    axes_train[0, 0].grid(True, alpha=0.3)

    # Derivative Norm
    axes_train[0, 1].plot(results['deriv_norms'], 'purple', linewidth=2)
    axes_train[0, 1].set_xlabel('Epoch')
    axes_train[0, 1].set_ylabel('Derivative L2 Norm')
    axes_train[0, 1].set_title('Derivative Norm (Non-Triviality Check)')
    axes_train[0, 1].grid(True, alpha=0.3)
    axes_train[0, 1].axhline(0.1, color='r', linestyle='--', alpha=0.5, label='Low norm warning')
    axes_train[0, 1].legend()

    # Learning Rate
    axes_train[1, 0].plot(results['lrs'], 'green', linewidth=2)
    axes_train[1, 0].set_xlabel('Epoch')
    axes_train[1, 0].set_ylabel('Learning Rate')
    axes_train[1, 0].set_title('Learning Rate Schedule')
    axes_train[1, 0].set_yscale('log')
    axes_train[1, 0].grid(True, alpha=0.3)

    # Total Loss (last 50%)
    start_idx = int(0.5 * len(results['losses']))
    axes_train[1, 1].plot(range(start_idx, len(results['losses'])), results['losses'][start_idx:], 'orange', linewidth=2)
    axes_train[1, 1].set_xlabel('Epoch')
    axes_train[1, 1].set_ylabel('Total Loss')
    axes_train[1, 1].set_title('Total Loss (Last 50%)')
    axes_train[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_train
    return axes_train, fig_train, start_idx


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
    ax_phase.set_title('Phase Portrait: True vs Neural ODE (Tuned)', fontsize=14, fontweight='bold')
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

    # Additional metrics
    final_deriv_norm = results['deriv_norms'][-1]
    final_lr = results['lrs'][-1]

    metrics_table = mo.ui.table([
        {"Metric": "Angle MAE", "Value": f"{mae_angle:.4e}"},
        {"Metric": "Velocity MAE", "Value": f"{mae_velocity:.4e}"},
        {"Metric": "Angle RMSE", "Value": f"{rmse_angle:.4e}"},
        {"Metric": "Velocity RMSE", "Value": f"{rmse_velocity:.4e}"},
        {"Metric": "Angle Max Error", "Value": f"{max_error_angle:.4e}"},
        {"Metric": "Velocity Max Error", "Value": f"{max_error_velocity:.4e}"},
        {"Metric": "Final MSE Loss", "Value": f"{results['mse_losses'][-1]:.4e}"},
        {"Metric": "Final Deriv Norm", "Value": f"{final_deriv_norm:.4e}"},
        {"Metric": "Final Learning Rate", "Value": f"{final_lr:.4e}"}
    ], selection=None)

    metrics_table
    return (
        final_deriv_norm,
        final_lr,
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

    **Tuned Neural ODE Improvements:**

    ✅ **Data Normalization:** Stabilizes training by handling different scales
    ✅ **He Initialization:** Prevents vanishing gradients in ReLU networks
    ✅ **Weight Decay:** L2 regularization prevents overfitting
    ✅ **LR Scheduler:** Adaptive learning rate improves convergence
    ✅ **Derivative Penalty:** Avoids trivial $\dot{x} = 0$ solutions
    ✅ **Gradient Clipping:** Prevents gradient explosion during BPTT

    **Key Insight:** Check derivative norm! If it's too small (<0.1), the network likely learned a trivial solution. The penalty term forces meaningful dynamics.

    **Performance Comparison:** Run both vanilla and tuned versions with same seed to see the difference!
    """)
    return


if __name__ == "__main__":
    app.run()
