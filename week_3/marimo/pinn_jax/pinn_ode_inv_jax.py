# /// script
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.19.9"
app = marimo.App(app_title="Inverse PINN ODE (JAX)", auto_download=["html"])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inverse Physics-Informed Neural Networks for ODEs (JAX)
    ## Learning Unknown Physical Parameters from Data

    **Inverse PINNs with JAX** combine:
    - **Equinox** neural networks with learnable parameters
    - **JAX autodiff** for computing derivatives
    - **Optax** optimizers for joint parameter/network training
    - **Functional programming** style with explicit PRNG handling

    | Forward PINNs | Inverse PINNs (JAX) |
    |--------------|---------------------|
    | Known physics parameters | **Learn** β alongside network weights |
    | Solution from equations | Solution + **parameter estimation** |
    | Physics loss only | Physics + **data misfit** loss |
    | No measurements needed | Uses **sparse observations** |

    **JAX Advantages:** JIT compilation, automatic vectorization, functional purity

    **Navigation:** [Jump to Control Panel](#control-panel)
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax

    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        solve_ivp = None
    try:
        import matplotlib.animation as animation
    except ImportError:
        animation = None
    return animation, eqx, jax, jnp, mo, np, optax, plt, solve_ivp


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 1. The Physical System

    The damped pendulum is governed by:

    $$\frac{d^2\theta}{dt^2} + \beta \frac{d\theta}{dt} + \frac{g}{l}\sin(\theta) = 0$$

    **Known parameters:** $g$ = gravity (9.81 m/s²), $l$ = length (m)

    **Unknown parameter:** $\beta$ = damping coefficient (**to be learned**)

    **Initial Conditions:** $\theta(0) = \theta_0$, $\dot{\theta}(0) = \omega_0$

    **Inverse Problem:** Given sparse measurements $\{(t_i, \theta_i)\}$, estimate $\beta$ while learning $\theta(t)$.
    """)
    return


@app.cell
def _(mo, np, solve_ivp):
    @mo.cache
    def reference_solution(t_min, t_max, u0, v0, beta_true, g, l, n_eval):
        """Numerical solution using scipy for comparison"""
        if solve_ivp is None:
            return None, None

        def ode(t, y):
            theta, omega = y
            return [omega, -beta_true * omega - (g / l) * np.sin(theta)]

        sol = solve_ivp(ode, (t_min, t_max), [u0, v0], dense_output=True)
        t_ref = np.linspace(t_min, t_max, n_eval)
        u_ref = sol.sol(t_ref)[0]
        return t_ref, u_ref

    return (reference_solution,)


@app.cell
def _(jax, mo, np, solve_ivp):
    @mo.cache
    def generate_measurements(t_min, t_max, u0, v0, beta_true, g, l, num_meas, noise_std, seed):
        """Generate synthetic measurement data with noise"""
        if solve_ivp is None:
            return None, None

        # Generate true solution
        def ode(t, y):
            theta, omega = y
            return [omega, -beta_true * omega - (g / l) * np.sin(theta)]

        sol = solve_ivp(ode, (t_min, t_max), [u0, v0], dense_output=True)

        # Sample measurement times using JAX random
        key = jax.random.PRNGKey(seed)
        t_meas = jax.random.uniform(key, (num_meas,), minval=t_min, maxval=t_max)
        t_meas = np.sort(np.array(t_meas))

        # Add noise to measurements
        rng = np.random.default_rng(seed)
        u_clean = sol.sol(t_meas)[0]
        u_meas = u_clean + rng.normal(0.0, noise_std, size=num_meas)

        return t_meas, u_meas

    return (generate_measurements,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 2. The Inverse PINN Formulation

    ### Core Idea

    Minimize a combined loss:

    $$\mathcal{L}_{\text{total}} = w_{\text{phys}} \mathcal{L}_{\text{physics}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{data}} \mathcal{L}_{\text{data}}$$

    where $\beta$ is a **learnable JAX array** alongside network weights.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.1 Equinox Network with Learnable Parameter

    The Equinox module includes:
    - Standard layers mapping $t \to \theta(t)$
    - **Learnable parameter** `beta_hat: jax.Array` (initialized to small value)
    """)
    return


@app.cell
def _(eqx, jax, jnp):
    class InversePINN(eqx.Module):
        layers: list
        beta_hat: jax.Array  # Learnable parameter

        def __init__(self, key, hidden_width=32, num_layers=2, beta_init=0.1):
            keys = jax.random.split(key, num_layers + 1)
            self.layers = [eqx.nn.Linear(1, hidden_width, key=keys[0])]
            for i in range(1, num_layers):
                self.layers.append(eqx.nn.Linear(hidden_width, hidden_width, key=keys[i]))
            self.layers.append(eqx.nn.Linear(hidden_width, 1, key=keys[num_layers]))
            self.beta_hat = jnp.array(beta_init)

        def __call__(self, t):
            h = t
            for layer in self.layers[:-1]:
                h = jax.nn.tanh(layer(h))
            return self.layers[-1](h)

    return (InversePINN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 Physics Residual

    Uses the **estimated** $\hat{\beta}$ in the ODE residual:

    $$\mathcal{L}_{\text{physics}} = \frac{1}{N}\sum_{i=1}^N \left[\frac{\partial^2 u}{\partial t^2} + \hat{\beta} \frac{\partial u}{\partial t} + \frac{g}{l}\sin(u)\right]^2$$
    """)
    return


@app.cell
def _(jax, jnp):
    def physics_residual(model, t_val, g, l):
        """Compute ODE residual for a single time point using model's beta_hat"""
        def u_fn(t):
            return model(jnp.array([[t]]))[0, 0]

        u = u_fn(t_val)
        u_t = jax.grad(u_fn)(t_val)
        u_tt = jax.grad(jax.grad(u_fn))(t_val)

        beta = model.beta_hat
        ode_res = u_tt + beta * u_t + (g / l) * jnp.sin(u)
        return ode_res

    def physics_residual_batch(model, t_batch, g, l):
        """Vectorized physics residual for batch of time points"""
        return jax.vmap(lambda t: physics_residual(model, t[0], g, l))(t_batch)

    return physics_residual, physics_residual_batch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.3 Data Misfit Loss

    Penalize deviation from measurements:

    $$\mathcal{L}_{\text{data}} = \frac{1}{M}\sum_{j=1}^M |u(t_j^{\text{meas}}) - u_j^{\text{meas}}|^2$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.4 Training Loop

    Jointly optimizes network weights and $\hat{\beta}$ via Optax optimizers with JIT compilation.
    """)
    return


@app.cell
def _(
    InversePINN,
    eqx,
    generate_measurements,
    jax,
    jnp,
    mo,
    np,
    optax,
    physics_residual_batch,
    reference_solution,
):
    @mo.persistent_cache
    def train_model(t_min, t_max, u0, v0, g, l, beta_true, beta_init,
                    n_collocation, num_meas, noise_std, meas_seed,
                    epochs, lr, physics_weight, ic_weight, data_weight,
                    hidden_width, num_layers, print_every, frame_every, make_gif, seed):
        # Generate measurement data
        t_meas_np, u_meas_np = generate_measurements(t_min, t_max, u0, v0, beta_true, g, l,
                                                      num_meas, noise_std, meas_seed)

        if t_meas_np is None:
            raise ValueError("Measurement generation failed (scipy not available)")

        t_meas = jnp.array(t_meas_np).reshape(-1, 1)
        u_meas = jnp.array(u_meas_np).reshape(-1, 1)

        # Initialize model
        key = jax.random.PRNGKey(seed)
        key, subkey = jax.random.split(key)
        model = InversePINN(subkey, hidden_width=hidden_width, num_layers=num_layers, beta_init=beta_init)

        # Setup optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

        # Collocation points
        key, subkey = jax.random.split(key)
        t_col = jax.random.uniform(subkey, (n_collocation,), minval=t_min, maxval=t_max).reshape(-1, 1)

        # Initial condition points
        t_ic = jnp.array([[t_min]])
        u_ic = jnp.array([[u0]])
        v_ic = jnp.array([[v0]])

        # Test points
        t_test = jnp.linspace(t_min, t_max, 500).reshape(-1, 1)

        # Reference solution
        t_ref, u_ref = reference_solution(t_min, t_max, u0, v0, beta_true, g, l, 500)

        # Loss function
        def loss_fn(model, t_col_pts, t_ic_val, u_ic_val, v_ic_val, t_meas_val, u_meas_val):
            # Physics loss
            ode_residuals = physics_residual_batch(model, t_col_pts, g, l)
            phys_loss = jnp.mean(ode_residuals ** 2)

            # IC loss
            u_pred_ic = model(t_ic_val)[0, 0]

            def u_fn_ic(t):
                return model(jnp.array([[t]]))[0, 0]

            u_t_pred_ic = jax.grad(u_fn_ic)(t_ic_val[0, 0])

            ic_loss = (u_pred_ic - u_ic_val[0, 0]) ** 2 + (u_t_pred_ic - v_ic_val[0, 0]) ** 2

            # Data misfit loss
            u_pred_meas = jax.vmap(lambda t: model(t)[0])(t_meas_val).reshape(-1, 1)
            data_loss = jnp.mean((u_pred_meas - u_meas_val) ** 2)

            # Total weighted loss
            total_loss = physics_weight * phys_loss + ic_weight * ic_loss + data_weight * data_loss

            return total_loss, (phys_loss, ic_loss, data_loss)

        # JIT-compiled training step
        @eqx.filter_jit
        def train_step(model, opt_state):
            (loss, (phys_loss, ic_loss_val, data_loss_val)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                model, t_col, t_ic, u_ic, v_ic, t_meas, u_meas
            )
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, phys_loss, ic_loss_val, data_loss_val

        animation_snapshots = []

        # Training loop
        losses = []
        beta_history = []

        for epoch in range(1, epochs + 1):
            model, opt_state, total_loss, phys_loss, ic_loss, data_loss = train_step(model, opt_state)

            losses.append([float(total_loss), float(phys_loss), float(ic_loss), float(data_loss)])
            beta_history.append(float(model.beta_hat))

            if epoch % print_every == 0:
                print(f"Epoch {epoch}/{epochs} | Loss={total_loss:.3e} | "
                      f"Phys={phys_loss:.3e} | IC={ic_loss:.3e} | Data={data_loss:.3e} | "
                      f"β̂={float(model.beta_hat):.5f}")

            if epoch % frame_every == 0:
                u_snapshot = jax.vmap(lambda t: model(t)[0])(t_test)
                animation_snapshots.append(
                    {
                        "epoch": epoch,
                        "u_pred": np.array(u_snapshot).flatten(),
                        "beta_hat": float(model.beta_hat),
                    }
                )

        # Final predictions
        u_pred = jax.vmap(lambda t: model(t)[0])(t_test)

        return {
            'model': model,
            'losses': np.array(losses),
            'beta_history': np.array(beta_history),
            't_test': np.array(t_test).flatten(),
            'u_pred': np.array(u_pred).flatten(),
            't_col': np.array(t_col).flatten(),
            't_meas': t_meas_np,
            'u_meas': u_meas_np,
            't_ref': t_ref,
            'u_ref': u_ref,
            'animation_snapshots': animation_snapshots,
            'make_gif': make_gif
        }

    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    <a id="control-panel"></a>
    ### 2.5 Interactive Parameters

    Adjust physical, measurement, and training parameters below. Results are cached.

    **Tuning tips:**
    - Keep physical parameters fixed unless the exercise explicitly asks you to change them.
    - Increase collocation points to enforce physics better; this usually improves accuracy but increases runtime.
    - Increase epochs while losses are still decreasing; stop when improvement plateaus.
    - Lower learning rate if training is unstable/oscillatory; raise it slightly if convergence is too slow.
    - Rebalance loss weights when one term dominates (physics, IC, or data loss).
    - Increase network width/depth for harder dynamics; larger models are slower and harder to optimize.
    """)
    return


@app.cell(hide_code=True)
def _(mo, np):
    # Physical parameters (known)
    g_slider = mo.ui.slider(5.0, 15.0, value=9.81, step=0.1, label="Gravity g (m/s²)")
    l_slider = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="Length l (m)")

    # Unknown parameter (true value for synthetic data)
    beta_true_slider = mo.ui.slider(0.1, 2.0, value=0.5, step=0.05, label="True damping β (for data gen)")

    # Initial parameter estimate
    beta_init_slider = mo.ui.slider(0.01, 1.0, value=0.1, step=0.01, label="Initial β̂ guess")

    # Initial conditions
    u0_slider = mo.ui.slider(0.0, 3.14, value=float(np.pi/2), step=0.1, label="Initial angle θ₀ (rad)")
    v0_slider = mo.ui.slider(-2.0, 2.0, value=0.0, step=0.1, label="Initial velocity ω₀ (rad/s)")

    # Time domain
    t_max_slider = mo.ui.slider(5.0, 20.0, value=10.0, step=1.0, label="Time horizon T (s)")
    n_collocation = mo.ui.slider(100, 1000, value=500, step=50, label="Collocation points")

    # Measurement settings
    num_meas_slider = mo.ui.slider(10, 200, value=50, step=10, label="Number of measurements")
    noise_slider = mo.ui.slider(0.0, 0.1, value=0.02, step=0.005, label="Measurement noise (std)")
    meas_seed_slider = mo.ui.slider(0, 9999, value=0, step=1, label="Measurement seed")

    # Training settings
    epochs_dropdown = mo.ui.dropdown({"5k": 5000, "10k": 10000, "20k": 20000, "30k": 30000},
                                      value="20k", label="Epochs")
    lr_dropdown = mo.ui.dropdown({"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "5e-3": 5e-3},
                                  value="1e-3", label="Learning rate")

    # Loss weights
    physics_weight = mo.ui.slider(0.0, 10.0, value=1.0, step=0.1, label="Physics loss weight")
    ic_weight = mo.ui.slider(0.0, 10.0, value=1.0, step=0.1, label="IC loss weight")
    data_weight_slider = mo.ui.slider(0.0, 20.0, value=5.0, step=0.5, label="Data loss weight")

    # Network architecture
    hidden_width_slider = mo.ui.slider(16, 128, value=32, step=8, label="Hidden layer width")
    num_layers_slider = mo.ui.slider(1, 5, value=2, step=1, label="Number of hidden layers")

    # Visualization
    make_gif_checkbox = mo.ui.checkbox(label="Show training animation", value=True)
    frame_interval = mo.ui.slider(50, 500, value=200, step=50, label="Animation frame interval")

    # JAX-specific: random seed
    seed_slider = mo.ui.slider(0, 9999, value=0, step=1, label="Random seed")

    control_panel = mo.vstack([
        mo.md("#### Known Physical Parameters"),
        g_slider, l_slider,
        mo.md("#### Unknown Parameter (Inverse Problem)"),
        beta_true_slider, beta_init_slider,
        mo.md("#### Initial Conditions"),
        u0_slider, v0_slider,
        mo.md("#### Domain & Training"),
        t_max_slider, n_collocation, epochs_dropdown, lr_dropdown,
        mo.md("#### Network Architecture"),
        hidden_width_slider, num_layers_slider,
        mo.md("#### Measurement Data"),
        num_meas_slider, noise_slider, meas_seed_slider,
        mo.md("#### Loss Weights"),
        physics_weight, ic_weight, data_weight_slider,
        mo.md("#### Visualization & JAX Settings"),
        make_gif_checkbox, frame_interval, seed_slider,
    ])

    train_button = mo.ui.run_button(label="▶ Train Inverse PINN (JAX)")

    mo.vstack([train_button, control_panel])
    return (
        beta_init_slider,
        beta_true_slider,
        data_weight_slider,
        epochs_dropdown,
        frame_interval,
        g_slider,
        hidden_width_slider,
        ic_weight,
        l_slider,
        lr_dropdown,
        make_gif_checkbox,
        meas_seed_slider,
        n_collocation,
        noise_slider,
        num_layers_slider,
        num_meas_slider,
        physics_weight,
        seed_slider,
        t_max_slider,
        train_button,
        u0_slider,
        v0_slider,
    )


@app.cell(hide_code=True)
def _(
    beta_init_slider,
    beta_true_slider,
    data_weight_slider,
    epochs_dropdown,
    frame_interval,
    g_slider,
    hidden_width_slider,
    ic_weight,
    l_slider,
    lr_dropdown,
    make_gif_checkbox,
    meas_seed_slider,
    mo,
    n_collocation,
    noise_slider,
    num_layers_slider,
    num_meas_slider,
    physics_weight,
    seed_slider,
    t_max_slider,
    train_button,
    train_model,
    u0_slider,
    v0_slider,
):
    mo.stop(not train_button.value, mo.md("_Click **▶ Train Inverse PINN (JAX)** to begin_"))

    results = train_model(
        t_min=0.0, t_max=t_max_slider.value,
        u0=u0_slider.value, v0=v0_slider.value,
        g=g_slider.value, l=l_slider.value,
        beta_true=beta_true_slider.value, beta_init=beta_init_slider.value,
        n_collocation=n_collocation.value,
        num_meas=num_meas_slider.value, noise_std=noise_slider.value, meas_seed=meas_seed_slider.value,
        epochs=epochs_dropdown.value, lr=lr_dropdown.value,
        physics_weight=physics_weight.value, ic_weight=ic_weight.value, data_weight=data_weight_slider.value,
        hidden_width=hidden_width_slider.value, num_layers=num_layers_slider.value,
        print_every=200, frame_every=frame_interval.value,
        make_gif=make_gif_checkbox.value, seed=seed_slider.value
    )
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---
    ## 3. Results Analysis

    Analyze parameter convergence, solution accuracy, and data fit.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.1 Solution with Measurements
    """)
    return


@app.cell(hide_code=True)
def _(animation, beta_true_slider, mo, np, plt, results, u0_slider):
    snapshots = results.get("animation_snapshots", [])
    if snapshots and animation is not None:
        try:
            _fig_anim, _ax_anim = plt.subplots(figsize=(12, 6))
            t_test = np.asarray(results["t_test"])
            t_meas = np.asarray(results["t_meas"])
            u_meas = np.asarray(results["u_meas"])
            t_col = np.asarray(results["t_col"])

            curve_arrays = [np.asarray(results["u_pred"])]
            if results["u_ref"] is not None:
                curve_arrays.append(np.asarray(results["u_ref"]))
            curve_arrays.extend(np.asarray(frame["u_pred"]) for frame in snapshots)
            all_values = np.concatenate(curve_arrays)
            y_min = float(all_values.min())
            y_max = float(all_values.max())
            pad = max(0.1 * (y_max - y_min), 1e-3)

            _ax_anim.set_xlim(float(t_test.min()), float(t_test.max()))
            _ax_anim.set_ylim(y_min - pad, y_max + pad)
            _ax_anim.set_xlabel("Time (s)", fontsize=12)
            _ax_anim.set_ylabel("Angle θ(t) (rad)", fontsize=12)
            _ax_anim.grid(True, alpha=0.3)

            _line_solution, = _ax_anim.plot([], [], "r-", linewidth=2, label="PINN")
            _ax_anim.scatter(
                t_meas,
                u_meas,
                s=35,
                c="green",
                alpha=0.6,
                label=f"Measurements (n={len(t_meas)})",
                edgecolors="darkgreen",
            )
            if results["u_ref"] is not None:
                _ax_anim.plot(results["t_ref"], results["u_ref"], "b--", linewidth=2, alpha=0.8, label="True solution")
            _ax_anim.scatter(t_col, np.zeros_like(t_col), s=10, c="orange", alpha=0.3, label="Collocation")
            _ax_anim.plot(0, u0_slider.value, "mo", markersize=8, label="IC")
            _epoch_text_solution = _ax_anim.text(0.02, 0.95, "", transform=_ax_anim.transAxes, va="top")
            _ax_anim.legend(loc="best")

            def _init_solution_anim():
                _line_solution.set_data([], [])
                _epoch_text_solution.set_text("")
                return _line_solution, _epoch_text_solution

            def _animate_solution_frame(i):
                frame = snapshots[i]
                _line_solution.set_data(t_test, frame["u_pred"])
                _epoch_text_solution.set_text(
                    f"Epoch: {frame['epoch']} | β̂={frame['beta_hat']:.4f} "
                    f"(true={beta_true_slider.value:.4f})"
                )
                return _line_solution, _epoch_text_solution

            _solution_anim_obj = animation.FuncAnimation(
                _fig_anim,
                _animate_solution_frame,
                init_func=_init_solution_anim,
                frames=len(snapshots),
                interval=170,
                blit=True,
            )
            _video_html_solution = _solution_anim_obj.to_html5_video()
            plt.close(_fig_anim)
            solution_output = mo.Html(_video_html_solution)
        except Exception as e:
            solution_output = mo.md(f"_Animation rendering error: {e}_")
    else:
        fig1, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(results["t_test"], results["u_pred"], "r-", linewidth=2, label="PINN", zorder=3)
        ax1.scatter(
            results["t_meas"],
            results["u_meas"],
            s=40,
            c="green",
            alpha=0.6,
            label=f"Measurements (n={len(results['t_meas'])})",
            zorder=4,
            edgecolors="darkgreen",
        )

        if results["u_ref"] is not None:
            ax1.plot(results["t_ref"], results["u_ref"], "b--", linewidth=2, label="True solution", alpha=0.8)
            error = np.abs(results["u_pred"] - results["u_ref"])
            ax1.fill_between(
                results["t_test"],
                results["u_pred"] - error,
                results["u_pred"] + error,
                alpha=0.2,
                color="red",
                label="Error",
            )

        ax1.scatter(
            results["t_col"],
            np.zeros_like(results["t_col"]),
            s=10,
            c="orange",
            alpha=0.3,
            label=f"Collocation (n={len(results['t_col'])})",
            zorder=2,
        )
        ax1.plot(0, u0_slider.value, "mo", markersize=10, label="IC", zorder=5)
        ax1.set_xlabel("Time (s)", fontsize=12)
        ax1.set_ylabel("Angle θ(t) (rad)", fontsize=12)
        ax1.set_title(
            f'Inverse PINN Solution (JAX) | β̂={float(results["model"].beta_hat):.4f} (true={beta_true_slider.value:.4f})',
            fontsize=14,
            fontweight="bold",
        )
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best")
        plt.tight_layout()
        solution_output = fig1

    solution_output
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.2 Parameter Convergence

    Track how $\hat{\beta}$ converges to the true value during training.
    """)
    return


@app.cell(hide_code=True)
def _(beta_true_slider, plt, results):
    fig_beta, ax_beta = plt.subplots(figsize=(12, 5))

    ax_beta.plot(results['beta_history'], 'purple', linewidth=2, label='β̂ (estimated)')
    ax_beta.axhline(beta_true_slider.value, color='blue', linestyle='--', linewidth=2,
                    label=f'β_true = {beta_true_slider.value:.4f}')
    ax_beta.axhline(float(results['model'].beta_hat), color='red', linestyle=':',
                    linewidth=1.5, alpha=0.7, label=f'Final β̂ = {float(results["model"].beta_hat):.4f}')

    ax_beta.set_xlabel('Epoch', fontsize=12)
    ax_beta.set_ylabel('Damping β', fontsize=12)
    ax_beta.set_title('Parameter Convergence (JAX)', fontsize=14, fontweight='bold')
    ax_beta.grid(True, alpha=0.3)
    ax_beta.legend()
    plt.tight_layout()
    fig_beta
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.3 Physics Residual

    The residual $\ddot{u} + \hat{\beta}\dot{u} + (g/l)\sin(u)$ should be $\approx 0$ everywhere.
    """)
    return


@app.cell(hide_code=True)
def _(
    g_slider,
    jax,
    jnp,
    l_slider,
    np,
    physics_residual,
    plt,
    results,
    t_max_slider,
):

    t_fine = jnp.linspace(0.0, t_max_slider.value, 1000).reshape(-1, 1)
    residual = jax.vmap(lambda t: physics_residual(results['model'], t[0], g_slider.value, l_slider.value))(t_fine)
    residual_np = np.array(residual).flatten()

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(np.array(t_fine).flatten(), residual_np, 'purple', linewidth=1.5)
    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.fill_between(np.array(t_fine).flatten(), 0, residual_np, alpha=0.3, color='purple')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('ODE Residual', fontsize=12)
    ax2.set_title(r'Physics Satisfaction: $\ddot{u} + \hat{\beta}\dot{u} + (g/l)\sin(u)$', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('symlog', linthresh=1e-6)
    plt.tight_layout()
    fig2
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.4 Training Dynamics
    """)
    return


@app.cell(hide_code=True)
def _(plt, results):
    losses = results['losses']

    fig3, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(14, 5))

    # Log scale
    ax_log.plot(losses[:, 0], 'k-', linewidth=2, label='Total', alpha=0.8)
    ax_log.plot(losses[:, 1], 'b-', linewidth=1.5, label='Physics', alpha=0.7)
    ax_log.plot(losses[:, 2], 'g-', linewidth=1.5, label='IC', alpha=0.7)
    ax_log.plot(losses[:, 3], 'm-', linewidth=1.5, label='Data', alpha=0.7)
    ax_log.set_yscale('log')
    ax_log.set_xlabel('Epoch')
    ax_log.set_ylabel('MSE Loss (log)')
    ax_log.set_title('Training Evolution (JAX)')
    ax_log.grid(True, alpha=0.3)
    ax_log.legend()

    # Linear scale (last 20%)
    start_idx = int(0.8 * len(losses))
    ax_lin.plot(range(start_idx, len(losses)), losses[start_idx:, 0], 'k-', linewidth=2, label='Total')
    ax_lin.plot(range(start_idx, len(losses)), losses[start_idx:, 1], 'b-', linewidth=1.5, label='Physics')
    ax_lin.plot(range(start_idx, len(losses)), losses[start_idx:, 2], 'g-', linewidth=1.5, label='IC')
    ax_lin.plot(range(start_idx, len(losses)), losses[start_idx:, 3], 'm-', linewidth=1.5, label='Data')
    ax_lin.set_xlabel('Epoch')
    ax_lin.set_ylabel('MSE Loss (linear)')
    ax_lin.set_title('Convergence Detail (Last 20%)')
    ax_lin.grid(True, alpha=0.3)
    ax_lin.legend()

    plt.tight_layout()
    fig3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.5 Error Metrics
    """)
    return


@app.cell(hide_code=True)
def _(
    beta_true_slider,
    g_slider,
    jax,
    jnp,
    l_slider,
    mo,
    np,
    physics_residual,
    results,
    t_max_slider,
):
    if results['u_ref'] is not None:
        mae = np.mean(np.abs(results['u_pred'] - results['u_ref']))
        max_error = np.max(np.abs(results['u_pred'] - results['u_ref']))
        rmse = np.sqrt(np.mean((results['u_pred'] - results['u_ref'])**2))

        t_metrics = jnp.linspace(0.0, t_max_slider.value, 1000).reshape(-1, 1)
        res_metrics = jax.vmap(lambda t: physics_residual(results['model'], t[0], g_slider.value,
                                                           l_slider.value))(t_metrics)
        residual_l2 = float(jnp.sqrt(jnp.mean(res_metrics**2)))

        # Parameter error
        beta_error = abs(float(results['model'].beta_hat) - beta_true_slider.value)
        beta_rel_error = beta_error / beta_true_slider.value * 100

        metrics_table = mo.ui.table([
            {"Metric": "Solution MAE", "Value": f"{mae:.4e}"},
            {"Metric": "Solution RMSE", "Value": f"{rmse:.4e}"},
            {"Metric": "Max Error", "Value": f"{max_error:.4e}"},
            {"Metric": "Physics Residual L2", "Value": f"{residual_l2:.4e}"},
            {"Metric": "β̂ Estimated", "Value": f"{float(results['model'].beta_hat):.6f}"},
            {"Metric": "β True", "Value": f"{beta_true_slider.value:.6f}"},
            {"Metric": "β Absolute Error", "Value": f"{beta_error:.6f}"},
            {"Metric": "β Relative Error", "Value": f"{beta_rel_error:.2f}%"},
            {"Metric": "Final Total Loss", "Value": f"{results['losses'][-1, 0]:.4e}"}
        ], selection=None)
    else:
        metrics_table = mo.md("_Reference solution unavailable_")

    metrics_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.6 Data Fit Quality

    Compare PINN predictions to measurements.
    """)
    return


@app.cell(hide_code=True)
def _(jax, jnp, np, plt, results):
    # Interpolate PINN predictions at measurement times
    t_meas_jax = jnp.array(results['t_meas']).reshape(-1, 1)
    u_pred_at_meas = jax.vmap(lambda t: results['model'](t)[0])(t_meas_jax)
    u_pred_at_meas_np = np.array(u_pred_at_meas).flatten()

    fig_data, (ax_scatter, ax_resid) = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot: predicted vs measured
    ax_scatter.scatter(results['u_meas'], u_pred_at_meas_np, alpha=0.6, s=50)
    ax_scatter.plot([results['u_meas'].min(), results['u_meas'].max()],
                    [results['u_meas'].min(), results['u_meas'].max()],
                    'r--', linewidth=2, label='Perfect fit')
    ax_scatter.set_xlabel('Measured θ', fontsize=12)
    ax_scatter.set_ylabel('Predicted θ', fontsize=12)
    ax_scatter.set_title('Measurement Fit Quality', fontsize=13)
    ax_scatter.grid(True, alpha=0.3)
    ax_scatter.legend()

    # Residual plot
    data_residuals = u_pred_at_meas_np - results['u_meas']
    ax_resid.scatter(results['t_meas'], data_residuals, alpha=0.6, s=50, c='purple')
    ax_resid.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax_resid.set_xlabel('Time (s)', fontsize=12)
    ax_resid.set_ylabel('Prediction - Measurement', fontsize=12)
    ax_resid.set_title('Data Residuals', fontsize=13)
    ax_resid.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_data
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3.7 Phase Portrait
    """)
    return


@app.cell(hide_code=True)
def _(jax, jnp, np, plt, results, t_max_slider, u0_slider, v0_slider):
    t_phase = jnp.linspace(0.0, t_max_slider.value, 500).reshape(-1, 1)

    def u_and_dudt(model, t_val):
        """Compute u and du/dt at a single time point"""
        def u_fn(t):
            return model(jnp.array([[t]]))[0, 0]
        u = u_fn(t_val[0])
        u_dot = jax.grad(u_fn)(t_val[0])
        return u, u_dot

    u_phase_vals, u_dot_vals = jax.vmap(lambda t: u_and_dudt(results['model'], t))(t_phase)

    fig4, ax4 = plt.subplots(figsize=(8, 8))
    ax4.plot(np.array(u_phase_vals), np.array(u_dot_vals),
            'purple', linewidth=2, label='PINN')

    if results['u_ref'] is not None:
        u_dot_ref = np.gradient(results['u_ref'], results['t_ref'])
        ax4.plot(results['u_ref'], u_dot_ref, 'b--', linewidth=2, alpha=0.7, label='True solution')

    ax4.plot(u0_slider.value, v0_slider.value, 'go', markersize=12, label='IC', zorder=5)
    ax4.plot(0, 0, 'rs', markersize=10, label='Equilibrium', zorder=5)
    ax4.set_xlabel('Angle θ (rad)', fontsize=12)
    ax4.set_ylabel('Velocity dθ/dt (rad/s)', fontsize=12)
    ax4.set_title('Phase Portrait (JAX)', fontsize=14)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.axhline(0, color='k', linestyle='-', alpha=0.2)
    ax4.axvline(0, color='k', linestyle='-', alpha=0.2)
    plt.tight_layout()
    fig4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 3.8 Animated Pendulum Comparison

    **Navy pendulum**: PINN prediction (red trace)
    **Blue pendulum**: Numerical solution (cyan trace)

    Watch how closely the two solutions match in real-time motion.
    """)
    return


@app.cell(hide_code=True)
def _(animation, l_slider, mo, np, plt, results):
    if animation is not None:
        try:
            fig_anim, ax_anim = plt.subplots(figsize=(8, 8))
            ax_anim.set_xlim(-1.5 * l_slider.value, 1.5 * l_slider.value)
            ax_anim.set_ylim(-1.5 * l_slider.value, 0.5 * l_slider.value)
            ax_anim.set_aspect('equal')
            ax_anim.set_title('Pendulum Motion: PINN vs Numerical (JAX)', fontsize=14)
            ax_anim.plot(0, 0, 'ko', markersize=8)

            # PINN pendulum (red/navy)
            line_pinn, = ax_anim.plot([], [], 'o-', linewidth=3, markersize=15,
                                      color='navy', label='PINN')
            trace_pinn, = ax_anim.plot([], [], '-', alpha=0.3, linewidth=1, color='red')

            # Numerical pendulum (blue/cyan) - only if available
            if results['u_ref'] is not None:
                line_num, = ax_anim.plot([], [], 'o-', linewidth=3, markersize=15,
                                        color='blue', alpha=0.7, label='Numerical')
                trace_num, = ax_anim.plot([], [], '-', alpha=0.3, linewidth=1, color='cyan')
                ax_anim.legend(loc='upper right')
                has_numerical = True
            else:
                line_num, trace_num = None, None
                has_numerical = False

            def init():
                line_pinn.set_data([], [])
                trace_pinn.set_data([], [])
                if has_numerical:
                    line_num.set_data([], [])
                    trace_num.set_data([], [])
                    return line_pinn, trace_pinn, line_num, trace_num
                return line_pinn, trace_pinn

            def animate(i):
                # PINN solution
                angle_pinn = results['u_pred'][i]
                x_pinn = l_slider.value * np.sin(angle_pinn)
                y_pinn = -l_slider.value * np.cos(angle_pinn)
                line_pinn.set_data([0, x_pinn], [0, y_pinn])

                start = max(0, i - 20)
                trace_x_pinn = l_slider.value * np.sin(results['u_pred'][start:i+1])
                trace_y_pinn = -l_slider.value * np.cos(results['u_pred'][start:i+1])
                trace_pinn.set_data(trace_x_pinn, trace_y_pinn)

                if has_numerical:
                    # Numerical solution
                    angle_num = results['u_ref'][i]
                    x_num = l_slider.value * np.sin(angle_num)
                    y_num = -l_slider.value * np.cos(angle_num)
                    line_num.set_data([0, x_num], [0, y_num])

                    trace_x_num = l_slider.value * np.sin(results['u_ref'][start:i+1])
                    trace_y_num = -l_slider.value * np.cos(results['u_ref'][start:i+1])
                    trace_num.set_data(trace_x_num, trace_y_num)
                    return line_pinn, trace_pinn, line_num, trace_num

                return line_pinn, trace_pinn

            anim = animation.FuncAnimation(fig_anim, animate, init_func=init,
                                          frames=len(results['u_pred']), interval=20, blit=True)

            # Convert to HTML5 video
            video_html = anim.to_html5_video()
            plt.close(fig_anim)

            animation_display = mo.Html(video_html)
        except Exception as e:
            animation_display = mo.md(f"_Animation error: {str(e)}_")
    else:
        animation_display = mo.md("_Install matplotlib with: `pip install matplotlib`_")

    animation_display
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## Summary

    **Inverse PINN with JAX:**
    - ✅ Learns **unknown parameters** from sparse data
    - ✅ **Equinox** modules with learnable `beta_hat: jax.Array`
    - ✅ **JIT compilation** via `@eqx.filter_jit` for fast training
    - ✅ **Automatic vectorization** with `jax.vmap`
    - ✅ **Functional programming** style with explicit PRNG key management

    **JAX-Specific Advantages:**
    - Composable transformations (`jax.grad`, `jax.vmap`, `jax.jit`)
    - Immutable data structures (functional purity)
    - Explicit randomness (PRNG key splitting)
    - XLA compilation for performance

    **Extensions:**
    - Multi-parameter estimation (β, g, l simultaneously)
    - Bayesian parameter inference with uncertainty quantification
    - Real experimental data integration
    """)
    return


if __name__ == "__main__":
    app.run()
