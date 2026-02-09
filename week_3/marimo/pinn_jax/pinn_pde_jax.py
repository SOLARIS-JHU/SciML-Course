# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "marimo",
#     "jax",
#     "equinox",
#     "optax",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "imageio",
#     "pillow",
# ]
#
# [tool.marimo.runtime]
# auto_instantiate = false
# on_cell_change = "lazy"
# ///

import marimo

__generated_with = "0.10.14"
app = marimo.App(app_title="PINN for 1D Heat Equation (JAX)")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Physics-Informed Neural Networks for PDEs
        ## Solving the 1D Heat Equation Without Discretization (JAX)

        **Physics-Informed Neural Networks (PINNs)** train neural networks to satisfy both governing equations
        and boundary/initial conditions simultaneously—no spatial or temporal discretization required.

        | Traditional PDE Methods | PINNs |
        |------------------------|-------|
        | Require spatial mesh | **Meshfree**: continuous solution everywhere |
        | Fixed grid resolution | **Adaptive**: query at any $(x,t)$ |
        | Discrete solutions | **Differentiable**: embed in optimization pipelines |
        | Struggle with sparse data | **Data-efficient**: fuse physics with observations |

        ---

        ## The Physical System

        The 1D heat equation models thermal diffusion:

        $$\frac{\partial u}{\partial t} = \alpha \frac{\partial^2 u}{\partial x^2}$$

        **Notation:**
        - $u(x,t)$ = temperature at position $x$ and time $t$
        - $\alpha$ = thermal diffusivity (controls diffusion rate, units: m²/s)

        **Physical Interpretation:** Heat flows from hot regions to cold regions, with rate proportional to the second spatial derivative (curvature). High curvature means rapid temperature change.

        ---

        ## Problem Setup

        **Domain:** $x \in [0, 1]$, $t \in [0, T]$

        **Initial Condition (IC):**

        $$u(x, 0) = \sin(\pi x)$$

        Initial temperature distribution is a half sine wave—hot in the middle, cool at edges.

        **Boundary Conditions (BC):**

        $$u(0, t) = 0, \quad u(1, t) = 0$$

        Both endpoints are held at zero temperature (Dirichlet boundary conditions).

        **Analytical Solution:** For this specific problem, the exact solution is known:

        $$u(x,t) = \sin(\pi x) e^{-\pi^2 \alpha t}$$

        This allows us to validate PINN accuracy against ground truth—the spatial profile decays exponentially in time.
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _():
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import optax
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from io import BytesIO
    from PIL import Image
    import imageio
    return (
        Axes3D,
        BytesIO,
        Image,
        cm,
        eqx,
        imageio,
        jax,
        jnp,
        np,
        optax,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 1. The PINN Formulation

        ### Core Idea

        Approximate $u(x,t)$ with neural network $u_\phi(x,t)$ by minimizing:

        $$\mathcal{L}_{\text{total}} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{BC}} \mathcal{L}_{\text{BC}}$$

        where derivatives are computed via **automatic differentiation**.

        ### 1.1 Physics Residual

        The governing equation enforces energy conservation at **collocation points** $\{(x_i, t_i)\}$:

        $$\mathcal{R}_{\text{PDE}}(x, t) = \frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0$$

        **Physics Loss:**

        $$\mathcal{L}_{\text{PDE}} = \frac{1}{N_c}\sum_{i=1}^{N_c} \left[\frac{\partial u}{\partial t}(x_i, t_i) - \alpha \frac{\partial^2 u}{\partial x^2}(x_i, t_i)\right]^2$$

        ### 1.2 Initial Condition

        Temperature distribution at $t=0$:

        $$\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}}\sum_{i=1}^{N_{\text{IC}}} |u(x_i, 0) - \sin(\pi x_i)|^2$$

        ### 1.3 Boundary Conditions

        Fixed temperature at boundaries (Dirichlet BCs):

        $$\mathcal{L}_{\text{BC}} = \frac{1}{N_{\text{BC}}}\sum_{i=1}^{N_{\text{BC}}} \left[|u(0, t_i)|^2 + |u(1, t_i)|^2\right]$$

        ### Why Multiple Loss Terms?

        - **PDE loss**: Ensures physics is satisfied throughout the domain
        - **IC loss**: Matches initial temperature profile
        - **BC loss**: Enforces boundary temperature constraints
        - **Weights** $w_{\text{PDE}}, w_{\text{IC}}, w_{\text{BC}}$: Balance competing objectives
        """
    )
    return


@app.cell
def _(np):
    def u_initial(x):
        """Initial condition: u(x, 0) = sin(pi*x)"""
        return np.sin(np.pi * x)

    def u_analytical(x, t, alpha):
        """Analytical solution: u(x,t) = sin(pi*x) * exp(-pi^2 * alpha * t)"""
        return np.sin(np.pi * x) * np.exp(-np.pi**2 * alpha * t)
    return u_analytical, u_initial





@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2. Neural Network Architecture

    A configurable feedforward network (tunable hidden layers and width, tanh activations) maps $(x, t)$ to temperature $u(x,t)$.

    **Input dimension:** 2 (position $x$ and time $t$)

    **Output dimension:** 1 (temperature $u$)

    **JAX/Equinox specifics:** Network is an Equinox module (pytree), compatible with JAX transformations like `jax.jit` and `jax.vmap`.
    """)
    return


@app.cell
def _(eqx, jax):
    class PINN(eqx.Module):
        """Physics-Informed Neural Network for 1D Heat Equation

        Architecture:
        - Input: (x, t) ∈ R²
        - Configurable hidden layers with adjustable width
        - Tanh activation functions
        - Output: u(x,t) ∈ R

        Implemented as Equinox module (JAX pytree)
        """
        layers: list

        def __init__(self, key, width=32, num_layers=3):
            keys = jax.random.split(key, num_layers + 1)
            self.layers = [eqx.nn.Linear(2, width, key=keys[0])]  # 2 inputs: x and t
            for i in range(1, num_layers):
                self.layers.append(eqx.nn.Linear(width, width, key=keys[i]))
            self.layers.append(eqx.nn.Linear(width, 1, key=keys[num_layers]))  # 1 output: u

        def __call__(self, xt):
            h = xt
            for layer in self.layers[:-1]:
                h = jax.nn.tanh(layer(h))
            return self.layers[-1](h)
    return (PINN,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 2.2 PDE Residual Computation via Automatic Differentiation

    JAX's `jax.grad` computes derivatives of the network output with respect to inputs.

    **Key steps:**

    1. **First-order derivatives**: Compute $\frac{\partial u}{\partial x}$ and $\frac{\partial u}{\partial t}$ using `jax.grad`
    2. **Second-order derivative**: Compute $\frac{\partial^2 u}{\partial x^2}$ by nesting `jax.grad` calls
    3. **Residual**: Combine as $\mathcal{R} = u_t - \alpha u_{xx}$
    4. **Vectorization**: Use `jax.vmap` to batch residual computation across collocation points

    **JAX functional style:** Pure functions with explicit argument passing (no side effects).
    """)
    return


@app.cell
def _(jax, jnp):
    def pde_residual_single(model, x_val, t_val, alpha):
        """Compute PDE residual for a single point (x, t)

        Uses JAX automatic differentiation:
        - jax.grad for first and second derivatives
        - Computes u_t, u_x, u_xx using nested gradients
        """
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

    def pde_residual_batch(model, x_batch, t_batch, alpha):
        """Vectorized PDE residual for batch of points using jax.vmap"""
        return jax.vmap(lambda x, t: pde_residual_single(model, x[0], t[0], alpha))(
            x_batch, t_batch
        )
    return pde_residual_batch, pde_residual_single


@app.cell
def _(eqx, jax, jnp, mo, np, optax):
    @mo.persistent_cache
    def train_model(
        x_min,
        x_max,
        t_min,
        t_max,
        alpha,
        n_collocation,
        n_ic,
        n_bc,
        epochs,
        lr,
        pde_weight,
        ic_weight,
        bc_weight,
        hidden_width,
        num_layers,
        print_every,
        frame_every,
        make_gif,
        seed,
    ):
        """Train PINN for 1D heat equation using JAX

        All UI parameters are function arguments for persistent caching.
        Returns dict with model, training history, and test predictions.
        """
        import matplotlib.pyplot as plt
        from io import BytesIO
        from PIL import Image

        # Import functions from outer scope
        from __main__ import PINN, pde_residual_batch, u_initial

        # Initialize PRNG key
        key = jax.random.PRNGKey(seed)

        # Generate collocation points (for PDE residual)
        key, subkey = jax.random.split(key)
        x_collocation = jax.random.uniform(subkey, (n_collocation, 1)) * (x_max - x_min) + x_min
        key, subkey = jax.random.split(key)
        t_collocation = jax.random.uniform(subkey, (n_collocation, 1)) * (t_max - t_min) + t_min
        collocation_points = jnp.concatenate([x_collocation, t_collocation], axis=1)

        # Generate initial condition points (IC)
        key, subkey = jax.random.split(key)
        x_ic = jax.random.uniform(subkey, (n_ic, 1)) * (x_max - x_min) + x_min
        t_ic = jnp.zeros((n_ic, 1))
        ic_points = jnp.concatenate([x_ic, t_ic], axis=1)
        u_ic = jnp.array(u_initial(np.array(x_ic)))

        # Generate boundary condition points (BC)
        key, subkey = jax.random.split(key)
        t_bc = jax.random.uniform(subkey, (n_bc, 1)) * (t_max - t_min) + t_min
        x_bc_left = jnp.full((n_bc, 1), x_min)
        x_bc_right = jnp.full((n_bc, 1), x_max)
        bc_points_left = jnp.concatenate([x_bc_left, t_bc], axis=1)
        bc_points_right = jnp.concatenate([x_bc_right, t_bc], axis=1)

        # Initialize model
        key, subkey = jax.random.split(key)
        pinn = PINN(subkey, width=hidden_width, num_layers=num_layers)

        # Define loss function
        def loss_fn(model, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right, alpha):
            """Compute total loss with three components"""
            # PDE Loss
            x_col = collocation_pts[:, 0:1]
            t_col = collocation_pts[:, 1:2]
            pde_res = pde_residual_batch(model, x_col, t_col, alpha)
            pde_loss = jnp.mean(pde_res ** 2)

            # IC Loss
            u_ic_pred = jax.vmap(lambda xt: model(xt)[0])(ic_pts).reshape(-1, 1)
            ic_loss = jnp.mean((u_ic_pred - u_ic_vals) ** 2)

            # BC Loss
            u_bc_pred_left = jax.vmap(lambda xt: model(xt)[0])(bc_pts_left).reshape(-1, 1)
            u_bc_pred_right = jax.vmap(lambda xt: model(xt)[0])(bc_pts_right).reshape(-1, 1)
            bc_loss = jnp.mean(u_bc_pred_left ** 2) + jnp.mean(u_bc_pred_right ** 2)

            # Total loss with weights
            total_loss = pde_weight * pde_loss + ic_weight * ic_loss + bc_weight * bc_loss

            return total_loss, (pde_loss, ic_loss, bc_loss)

        # Define JIT-compiled training step
        @eqx.filter_jit
        def train_step(model, opt_state, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right, alpha):
            """Single training step with gradient update"""
            (loss, (pde_loss, ic_loss, bc_loss)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
                model, collocation_pts, ic_pts, u_ic_vals, bc_pts_left, bc_pts_right, alpha
            )
            updates, opt_state = optimizer.update(grads, opt_state, model)
            model = eqx.apply_updates(model, updates)
            return model, opt_state, loss, pde_loss, ic_loss, bc_loss

        # Initialize optimizer
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(pinn, eqx.is_array))

        # Training loop
        losses = {
            'total': [],
            'pde': [],
            'ic': [],
            'bc': []
        }

        gif_frames = []

        for epoch in range(1, epochs + 1):
            pinn, opt_state, loss, pde_loss, ic_loss, bc_loss = train_step(
                pinn, opt_state, collocation_points, ic_points, u_ic,
                bc_points_left, bc_points_right, alpha
            )

            # Record losses (convert to Python floats)
            losses['total'].append(float(loss))
            losses['pde'].append(float(pde_loss))
            losses['ic'].append(float(ic_loss))
            losses['bc'].append(float(bc_loss))

            if epoch % print_every == 0:
                print(f"Epoch [{epoch}/{epochs}], Total Loss: {float(loss):.6f}, "
                      f"PDE: {float(pde_loss):.6f}, IC: {float(ic_loss):.6f}, BC: {float(bc_loss):.6f}")

            # Capture frames for GIF
            if make_gif and epoch % frame_every == 0:
                fig = plt.figure(figsize=(10, 4))

                # Left: Loss curves
                ax1 = fig.add_subplot(121)
                ax1.semilogy(losses['total'], label='Total', linewidth=2)
                ax1.semilogy(losses['pde'], label='PDE', alpha=0.7)
                ax1.semilogy(losses['ic'], label='IC', alpha=0.7)
                ax1.semilogy(losses['bc'], label='BC', alpha=0.7)
                ax1.set_xlabel('Epoch')
                ax1.set_ylabel('Loss (log scale)')
                ax1.set_title(f'Training Progress (Epoch {epoch})')
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                # Right: Current solution
                ax2 = fig.add_subplot(122)
                n_plot = 50
                x_plot = np.linspace(x_min, x_max, n_plot)
                t_plot = np.linspace(t_min, t_max, n_plot)
                X_plot, T_plot = np.meshgrid(x_plot, t_plot)
                xt_flat = jnp.concatenate([
                    jnp.array(X_plot.flatten()).reshape(-1, 1),
                    jnp.array(T_plot.flatten()).reshape(-1, 1)
                ], axis=1)

                u_pred_temp = jax.vmap(lambda xt: pinn(xt)[0])(xt_flat)
                u_pred_temp = np.array(u_pred_temp).reshape(n_plot, n_plot)

                im = ax2.contourf(X_plot, T_plot, u_pred_temp, levels=20, cmap='viridis')
                ax2.set_xlabel('x')
                ax2.set_ylabel('t')
                ax2.set_title('Current Solution u(x,t)')
                plt.colorbar(im, ax=ax2)

                plt.tight_layout()

                # Save to buffer
                buf = BytesIO()
                plt.savefig(buf, format='png', dpi=80, bbox_inches='tight')
                buf.seek(0)
                gif_frames.append(Image.open(buf).copy())
                buf.close()
                plt.close(fig)

        # Generate test predictions
        n_points = 100
        x_test = np.linspace(x_min, x_max, n_points)
        t_test = np.linspace(t_min, t_max, n_points)
        X_test, T_test = np.meshgrid(x_test, t_test)
        xt_test_flat = jnp.concatenate([
            jnp.array(X_test.flatten()).reshape(-1, 1),
            jnp.array(T_test.flatten()).reshape(-1, 1)
        ], axis=1)

        u_pred_test = jax.vmap(lambda xt: pinn(xt)[0])(xt_test_flat)
        u_pred_test = np.array(u_pred_test).reshape(n_points, n_points)

        # Compute analytical solution
        u_analytical_test = np.sin(np.pi * X_test) * np.exp(-np.pi**2 * alpha * T_test)

        # Convert collocation points to numpy
        collocation_np = np.array(collocation_points)

        # Compute PDE residuals at collocation points
        pde_res_final = pde_residual_batch(
            pinn,
            collocation_points[:, 0:1],
            collocation_points[:, 1:2],
            alpha
        )
        pde_res_final = np.array(pde_res_final)

        return {
            'model': pinn,
            'losses': losses,
            'x_test': x_test,
            't_test': t_test,
            'X_test': X_test,
            'T_test': T_test,
            'u_pred': u_pred_test,
            'u_analytical': u_analytical_test,
            'collocation_points': collocation_np,
            'pde_residuals': pde_res_final,
            'gif_frames': gif_frames,
            'make_gif': make_gif,
        }
    return (train_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        ## 3. Interactive Parameters & Training

        Adjust the sliders below to configure the PINN training:

        - **Physical parameters**: Thermal diffusivity, domain size
        - **Training parameters**: Collocation points, epochs, learning rate
        - **Loss weights**: Balance between PDE, IC, and BC enforcement
        - **Network architecture**: Hidden layer width
        - **JAX settings**: PRNG seed for reproducibility
        - **Visualization**: GIF generation options
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    # Physical parameters
    alpha_slider = mo.ui.slider(
        0.001, 0.1, value=0.01, step=0.001, label="Thermal diffusivity α"
    )
    x_max_slider = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="Domain length L")
    t_max_slider = mo.ui.slider(0.5, 2.0, value=1.0, step=0.1, label="Time horizon T")

    # Training parameters
    n_collocation_slider = mo.ui.slider(
        1000, 10000, value=5000, step=500, label="Collocation points"
    )
    n_ic_slider = mo.ui.slider(50, 300, value=100, step=10, label="IC points")
    n_bc_slider = mo.ui.slider(50, 300, value=100, step=10, label="BC points")
    epochs_dropdown = mo.ui.dropdown(
        {
            "5k": 5000,
            "10k": 10000,
            "20k": 20000,
            "40k": 40000,
            "60k": 60000,
        },
        value="40k",
        label="Epochs",
    )
    lr_dropdown = mo.ui.dropdown(
        {"1e-4": 1e-4, "5e-4": 5e-4, "1e-3": 1e-3, "2e-3": 2e-3},
        value="1e-3",
        label="Learning rate",
    )

    # Loss weights
    pde_weight_slider = mo.ui.slider(
        0.1, 10.0, value=1.0, step=0.1, label="PDE loss weight"
    )
    ic_weight_slider = mo.ui.slider(
        0.1, 20.0, value=10.0, step=0.5, label="IC loss weight"
    )
    bc_weight_slider = mo.ui.slider(
        0.1, 20.0, value=10.0, step=0.5, label="BC loss weight"
    )

    # Network architecture
    hidden_width_slider = mo.ui.slider(
        16, 64, value=32, step=8, label="Hidden layer width"
    )
    num_layers_slider = mo.ui.slider(
        1, 5, value=3, step=1, label="Number of hidden layers"
    )

    # Randomness
    seed_slider = mo.ui.slider(
        0, 9999, value=42, step=1, label="Random seed (PRNG key)"
    )

    # Visualization
    make_gif_checkbox = mo.ui.checkbox(
        label="Generate training GIF", value=False
    )
    frame_interval_slider = mo.ui.slider(
        50, 500, value=200, step=50, label="GIF frame interval (epochs)"
    )
    print_every_slider = mo.ui.slider(
        500, 5000, value=2000, step=500, label="Print interval (epochs)"
    )
    return (
        alpha_slider,
        bc_weight_slider,
        epochs_dropdown,
        frame_interval_slider,
        hidden_width_slider,
        ic_weight_slider,
        lr_dropdown,
        make_gif_checkbox,
        n_bc_slider,
        n_collocation_slider,
        n_ic_slider,
        num_layers_slider,
        pde_weight_slider,
        print_every_slider,
        seed_slider,
        t_max_slider,
        x_max_slider,
    )


@app.cell(hide_code=True)
def _(
    alpha_slider,
    bc_weight_slider,
    epochs_dropdown,
    frame_interval_slider,
    hidden_width_slider,
    ic_weight_slider,
    lr_dropdown,
    make_gif_checkbox,
    mo,
    n_bc_slider,
    n_collocation_slider,
    n_ic_slider,
    num_layers_slider,
    pde_weight_slider,
    print_every_slider,
    seed_slider,
    t_max_slider,
    x_max_slider,
):
    control_panel = mo.vstack([
        mo.md("#### Physical Parameters"),
        alpha_slider,
        x_max_slider,
        t_max_slider,
        mo.md("#### Domain Sampling"),
        n_collocation_slider,
        n_ic_slider,
        n_bc_slider,
        mo.md("#### Training Configuration"),
        epochs_dropdown,
        lr_dropdown,
        mo.md("#### Network Architecture"),
        hidden_width_slider,
        num_layers_slider,
        mo.md("#### JAX Settings"),
        seed_slider,
        mo.md("#### Loss Weights"),
        pde_weight_slider,
        ic_weight_slider,
        bc_weight_slider,
        mo.md("#### Visualization"),
        make_gif_checkbox,
        frame_interval_slider,
        print_every_slider,
    ])

    train_button = mo.ui.run_button(label="▶ Train PINN")

    mo.vstack([train_button, control_panel])
    return control_panel, train_button


@app.cell
def _(
    alpha_slider,
    bc_weight_slider,
    epochs_dropdown,
    frame_interval_slider,
    hidden_width_slider,
    ic_weight_slider,
    lr_dropdown,
    make_gif_checkbox,
    mo,
    n_bc_slider,
    n_collocation_slider,
    n_ic_slider,
    num_layers_slider,
    pde_weight_slider,
    print_every_slider,
    seed_slider,
    t_max_slider,
    train_button,
    train_model,
    x_max_slider,
):
    mo.stop(not train_button.value, mo.md("_Click **▶ Train PINN** to begin training_"))

    results = train_model(
        x_min=0.0,
        x_max=x_max_slider.value,
        t_min=0.0,
        t_max=t_max_slider.value,
        alpha=alpha_slider.value,
        n_collocation=n_collocation_slider.value,
        n_ic=n_ic_slider.value,
        n_bc=n_bc_slider.value,
        epochs=epochs_dropdown.value,
        lr=lr_dropdown.value,
        pde_weight=pde_weight_slider.value,
        ic_weight=ic_weight_slider.value,
        bc_weight=bc_weight_slider.value,
        hidden_width=hidden_width_slider.value,
        num_layers=num_layers_slider.value,
        print_every=print_every_slider.value,
        frame_every=frame_interval_slider.value,
        make_gif=make_gif_checkbox.value,
        seed=seed_slider.value,
    )

    mo.md(f"✅ **Training complete!** Final total loss: {results['losses']['total'][-1]:.6e}")
    return (results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    ## 4. Results Analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.1 3D Surface Plot: PINN Solution""")
    return


@app.cell(hide_code=True)
def _(Axes3D, cm, plt, results):
    fig_3d = plt.figure(figsize=(12, 8))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    surf = ax_3d.plot_surface(
        results['X_test'],
        results['T_test'],
        results['u_pred'],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True,
        alpha=0.9
    )
    ax_3d.set_title("PINN Solution: u(x, t)", fontsize=14, fontweight='bold')
    ax_3d.set_xlabel("Position x", fontsize=12)
    ax_3d.set_ylabel("Time t", fontsize=12)
    ax_3d.set_zlabel("Temperature u", fontsize=12)
    ax_3d.view_init(elev=25, azim=135)
    fig_3d.colorbar(surf, ax=ax_3d, shrink=0.5, aspect=10)
    plt.tight_layout()
    fig_3d
    return ax_3d, fig_3d, surf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.2 Comparison with Analytical Solution""")
    return


@app.cell(hide_code=True)
def _(Axes3D, cm, np, plt, results):
    fig_compare = plt.figure(figsize=(18, 5))

    # PINN Solution
    ax1 = fig_compare.add_subplot(131, projection='3d')
    surf1 = ax1.plot_surface(
        results['X_test'],
        results['T_test'],
        results['u_pred'],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True
    )
    ax1.set_title("PINN Solution (JAX)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("x")
    ax1.set_ylabel("t")
    ax1.set_zlabel("u")

    # Analytical Solution
    ax2 = fig_compare.add_subplot(132, projection='3d')
    surf2 = ax2.plot_surface(
        results['X_test'],
        results['T_test'],
        results['u_analytical'],
        cmap=cm.viridis,
        linewidth=0,
        antialiased=True
    )
    ax2.set_title("Analytical Solution", fontsize=12, fontweight='bold')
    ax2.set_xlabel("x")
    ax2.set_ylabel("t")
    ax2.set_zlabel("u")

    # Error
    error = np.abs(results['u_pred'] - results['u_analytical'])
    ax3 = fig_compare.add_subplot(133, projection='3d')
    surf3 = ax3.plot_surface(
        results['X_test'],
        results['T_test'],
        error,
        cmap=cm.plasma,
        linewidth=0,
        antialiased=True
    )
    ax3.set_title("Absolute Error", fontsize=12, fontweight='bold')
    ax3.set_xlabel("x")
    ax3.set_ylabel("t")
    ax3.set_zlabel("|error|")

    plt.tight_layout()
    fig_compare
    return ax1, ax2, ax3, error, fig_compare, surf1, surf2, surf3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.3 2D Heatmap Visualization""")
    return


@app.cell(hide_code=True)
def _(error, plt, results):
    fig_heatmap, axes = plt.subplots(1, 3, figsize=(18, 5))

    # PINN Solution
    im1 = axes[0].contourf(
        results['X_test'],
        results['T_test'],
        results['u_pred'],
        levels=30,
        cmap='viridis'
    )
    axes[0].set_title("PINN Solution u(x,t)", fontsize=12, fontweight='bold')
    axes[0].set_xlabel("Position x")
    axes[0].set_ylabel("Time t")
    plt.colorbar(im1, ax=axes[0])

    # Analytical Solution
    im2 = axes[1].contourf(
        results['X_test'],
        results['T_test'],
        results['u_analytical'],
        levels=30,
        cmap='viridis'
    )
    axes[1].set_title("Analytical Solution", fontsize=12, fontweight='bold')
    axes[1].set_xlabel("Position x")
    axes[1].set_ylabel("Time t")
    plt.colorbar(im2, ax=axes[1])

    # Error
    im3 = axes[2].contourf(
        results['X_test'],
        results['T_test'],
        error,
        levels=30,
        cmap='plasma'
    )
    axes[2].set_title("Absolute Error", fontsize=12, fontweight='bold')
    axes[2].set_xlabel("Position x")
    axes[2].set_ylabel("Time t")
    plt.colorbar(im3, ax=axes[2])

    plt.tight_layout()
    fig_heatmap
    return axes, fig_heatmap, im1, im2, im3


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.4 Temporal Slices: u(x) at Different Times""")
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    fig_time_slices, ax_time = plt.subplots(figsize=(12, 6))

    # Select 5 time slices
    t_slices = [0.1, 0.3, 0.5, 0.7, 0.9]
    t_max_val = results['t_test'].max()

    for t_val in t_slices:
        if t_val <= t_max_val:
            # Find closest time index
            t_idx = np.argmin(np.abs(results['t_test'] - t_val))

            # PINN solution
            ax_time.plot(
                results['x_test'],
                results['u_pred'][t_idx, :],
                label=f't={t_val:.1f} (PINN)',
                linewidth=2
            )

            # Analytical solution
            ax_time.plot(
                results['x_test'],
                results['u_analytical'][t_idx, :],
                '--',
                label=f't={t_val:.1f} (Analytical)',
                alpha=0.7
            )

    ax_time.set_xlabel("Position x", fontsize=12)
    ax_time.set_ylabel("Temperature u", fontsize=12)
    ax_time.set_title("Temperature Distribution at Different Times", fontsize=14, fontweight='bold')
    ax_time.legend(ncol=2, fontsize=9)
    ax_time.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_time_slices
    return ax_time, fig_time_slices, t_idx, t_max_val, t_slices, t_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.5 Spatial Slices: u(t) at Different Positions""")
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    fig_space_slices, ax_space = plt.subplots(figsize=(12, 6))

    # Select 3 spatial positions
    x_max_val = results['x_test'].max()
    x_slices = [0.25 * x_max_val, 0.5 * x_max_val, 0.75 * x_max_val]

    for x_val in x_slices:
        # Find closest position index
        x_idx = np.argmin(np.abs(results['x_test'] - x_val))

        # PINN solution
        ax_space.plot(
            results['t_test'],
            results['u_pred'][:, x_idx],
            label=f'x={x_val:.2f} (PINN)',
            linewidth=2
        )

        # Analytical solution
        ax_space.plot(
            results['t_test'],
            results['u_analytical'][:, x_idx],
            '--',
            label=f'x={x_val:.2f} (Analytical)',
            alpha=0.7
        )

    ax_space.set_xlabel("Time t", fontsize=12)
    ax_space.set_ylabel("Temperature u", fontsize=12)
    ax_space.set_title("Temperature Evolution at Different Positions", fontsize=14, fontweight='bold')
    ax_space.legend(ncol=2, fontsize=9)
    ax_space.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_space_slices
    return ax_space, fig_space_slices, x_idx, x_max_val, x_slices, x_val


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.6 Physics Residual Validation""")
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    fig_residual, axes_res = plt.subplots(1, 2, figsize=(16, 5))

    # Scatter plot of residuals
    residuals_flat = results['pde_residuals'].flatten()
    x_col = results['collocation_points'][:, 0]
    t_col = results['collocation_points'][:, 1]

    sc = axes_res[0].scatter(
        x_col,
        t_col,
        c=np.abs(residuals_flat),
        cmap='plasma',
        s=1,
        alpha=0.6
    )
    axes_res[0].set_xlabel("Position x", fontsize=12)
    axes_res[0].set_ylabel("Time t", fontsize=12)
    axes_res[0].set_title("PDE Residual Magnitude |R(x,t)|", fontsize=12, fontweight='bold')
    plt.colorbar(sc, ax=axes_res[0])

    # Histogram of residuals
    axes_res[1].hist(residuals_flat, bins=50, edgecolor='black', alpha=0.7)
    axes_res[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Zero residual')
    axes_res[1].set_xlabel("Residual value", fontsize=12)
    axes_res[1].set_ylabel("Frequency", fontsize=12)
    axes_res[1].set_title("Distribution of PDE Residuals", fontsize=12, fontweight='bold')
    axes_res[1].legend()
    axes_res[1].grid(True, alpha=0.3)

    plt.tight_layout()

    residual_stats = f"""
    **Residual Statistics:**
    - Mean: {np.mean(residuals_flat):.6e}
    - Std: {np.std(residuals_flat):.6e}
    - Max: {np.max(np.abs(residuals_flat)):.6e}
    - L2 norm: {np.linalg.norm(residuals_flat):.6e}
    """

    fig_residual
    return (
        axes_res,
        fig_residual,
        residual_stats,
        residuals_flat,
        sc,
        t_col,
        x_col,
    )


@app.cell(hide_code=True)
def _(mo, residual_stats):
    mo.md(residual_stats)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.7 Training Dynamics""")
    return


@app.cell(hide_code=True)
def _(np, plt, results):
    fig_training, axes_train = plt.subplots(2, 2, figsize=(16, 10))

    epochs_array = np.arange(1, len(results['losses']['total']) + 1)

    # Total loss (log scale)
    axes_train[0, 0].semilogy(epochs_array, results['losses']['total'], linewidth=2, color='black')
    axes_train[0, 0].set_xlabel("Epoch")
    axes_train[0, 0].set_ylabel("Total Loss (log scale)")
    axes_train[0, 0].set_title("Total Loss Evolution", fontweight='bold')
    axes_train[0, 0].grid(True, alpha=0.3)

    # Individual losses (log scale)
    axes_train[0, 1].semilogy(epochs_array, results['losses']['pde'], label='PDE', alpha=0.8)
    axes_train[0, 1].semilogy(epochs_array, results['losses']['ic'], label='IC', alpha=0.8)
    axes_train[0, 1].semilogy(epochs_array, results['losses']['bc'], label='BC', alpha=0.8)
    axes_train[0, 1].set_xlabel("Epoch")
    axes_train[0, 1].set_ylabel("Loss (log scale)")
    axes_train[0, 1].set_title("Loss Components (Log Scale)", fontweight='bold')
    axes_train[0, 1].legend()
    axes_train[0, 1].grid(True, alpha=0.3)

    # Total loss (linear scale)
    axes_train[1, 0].plot(epochs_array, results['losses']['total'], linewidth=2, color='black')
    axes_train[1, 0].set_xlabel("Epoch")
    axes_train[1, 0].set_ylabel("Total Loss")
    axes_train[1, 0].set_title("Total Loss Evolution (Linear Scale)", fontweight='bold')
    axes_train[1, 0].grid(True, alpha=0.3)

    # Individual losses (linear scale)
    axes_train[1, 1].plot(epochs_array, results['losses']['pde'], label='PDE', alpha=0.8)
    axes_train[1, 1].plot(epochs_array, results['losses']['ic'], label='IC', alpha=0.8)
    axes_train[1, 1].plot(epochs_array, results['losses']['bc'], label='BC', alpha=0.8)
    axes_train[1, 1].set_xlabel("Epoch")
    axes_train[1, 1].set_ylabel("Loss")
    axes_train[1, 1].set_title("Loss Components (Linear Scale)", fontweight='bold')
    axes_train[1, 1].legend()
    axes_train[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_training
    return axes_train, epochs_array, fig_training


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### 4.8 Error Metrics Summary""")
    return


@app.cell(hide_code=True)
def _(error, np, residuals_flat):
    mae = np.mean(error)
    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(error)
    residual_l2 = np.linalg.norm(residuals_flat)

    error_data = [
        ["Mean Absolute Error (MAE)", f"{mae:.6e}"],
        ["Root Mean Square Error (RMSE)", f"{rmse:.6e}"],
        ["Maximum Error", f"{max_error:.6e}"],
        ["PDE Residual L2 Norm", f"{residual_l2:.6e}"],
    ]
    return error_data, mae, max_error, residual_l2, rmse


@app.cell(hide_code=True)
def _(error_data, mo):
    mo.md(
        f"""
        | Metric | Value |
        |--------|-------|
        {chr(10).join(f"| {row[0]} | {row[1]} |" for row in error_data)}
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Summary

        ### Key Features of this JAX PINN Implementation

        ✅ **JAX Functional Programming**
        - Explicit PRNG key management for reproducibility
        - `jax.vmap` for efficient batched operations
        - `jax.grad` for automatic differentiation
        - `@eqx.filter_jit` for JIT-compiled training

        ✅ **Equinox Neural Networks**
        - Pytree-based module system
        - Clean functional design
        - Compatible with JAX transformations

        ✅ **Multi-term Loss Function**
        - PDE residual enforces heat equation at collocation points
        - Initial condition loss ensures correct starting temperature
        - Boundary condition losses maintain fixed boundary temperatures

        ✅ **Validation Against Analytical Solution**
        - Exact solution: $u(x,t) = \sin(\pi x) e^{-\pi^2 \alpha t}$
        - Provides ground truth for error analysis

        ### JAX vs PyTorch Comparison

        | Aspect | JAX | PyTorch |
        |--------|-----|---------|
        | **Programming style** | Functional (pure functions) | Object-oriented (classes) |
        | **Randomness** | Explicit PRNG keys | Global random state |
        | **Derivatives** | `jax.grad` + nested calls | `torch.autograd.grad` with `create_graph=True` |
        | **Batching** | `jax.vmap` (explicit) | Broadcasting (implicit) |
        | **Device** | Automatic placement | Manual `.to(device)` |
        | **JIT** | `@jax.jit` decorator | `torch.jit.script` |

        ### Strengths

        - **No mesh required**: Unlike finite difference/element methods
        - **Smooth solutions**: Neural network provides continuous interpolation
        - **Physics-informed**: Satisfies governing equations during training
        - **JAX performance**: Fast execution via JIT compilation

        ### Limitations

        - **Computational cost**: Training can be slow for complex PDEs
        - **Hyperparameter sensitivity**: Loss weights affect convergence
        - **Local minima**: May require multiple training runs
        - **Memory usage**: JAX traces functions for JIT

        ### Extensions

        - 2D/3D heat equations
        - Time-dependent boundary conditions
        - Nonlinear PDEs (reaction-diffusion, Burgers' equation)
        - Inverse problems (learning unknown parameters like α)
        """
    )
    return


if __name__ == "__main__":
    app.run()
