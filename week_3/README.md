# Week 3: Physics-Informed Neural Networks (PINNs)

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-thick.svg">
    <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="Marimo" height="50" />
  </picture>
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="50" />
  &nbsp;&nbsp;&nbsp;
  <img src="https://github.com/google/jax/raw/main/images/jax_logo_250px.png" alt="JAX" height="50" />
</p>

Week 3 course materials for experimenting with PINNs in PyTorch and JAX, with a focus on tuning model and training hyperparameters so the networks satisfy the target differential equations more accurately.

## Interactive Notebooks

### <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="marimo" height="40" /> +<img src="https://raw.githubusercontent.com/pytorch/pytorch/main/docs/source/_static/img/pytorch-logo-dark.png" alt="PyTorch" height="30" />  

- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_HcPEzSg3i34wwAAxCm1Ms6) Physics-Informed Neural Networks for ODEs
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_msQFS4Lz7i1RkyamB4wonn) Inverse Physics-Informed Neural Networks for ODEs
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_KgVPsqWowy4t6fic4AW2KT) Physics-Informed Neural Networks for PDEs

### <img src="https://github.com/marimo-team/marimo/raw/main/docs/_static/marimo-logotype-horizontal.png" alt="marimo" height="40" /> + <img src="https://github.com/google/jax/raw/main/images/jax_logo_250px.png" alt="JAX" height="30" /> 

- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_p93Ja6m3TatJiX5qu2c9Zq) Physics-Informed Neural Networks for ODEs
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_Jo8Jpnb71ERjrKywc9uvG6) Inverse Physics-Informed Neural Networks for ODEs
- [![Open in molab](https://molab.marimo.io/molab-shield.svg)](https://molab.marimo.io/notebooks/nb_iTvYExwNeGC1SzwQBQXUoJ) Physics-Informed Neural Networks for PDEs

## Local Run (Optional)

If the online notebooks feel slow on the server, run the notebooks with local host or use the plain Python scripts in `week_3/scripts`. See examples below: 

```bash
# Option 1: marimo app
marimo run week_3/marimo/pinn_torch/pinn_ode_torch.py 

# Option 2: marimo editor mode
marimo edit week_3/marimo/pinn_torch/pinn_ode_torch.py

# Option3: script fallback
python week_3/scripts/pinn_torch/PINN_ODE_torch.py
```

## Week 3 Structure Map

```text
week_3/
├── README.md                           # Week 3 landing page (this file)
├── Week_3_pinns.pptx                   # Lecture slides
├── PINN training strategies.pdf        # Reading/reference notes
├── marimo/                             # Interactive notebook-style apps
│   ├── pinn_torch/                     # PyTorch + marimo apps
│   │   ├── pinn_ode_torch.py           # Forward PINN (ODE)
│   │   ├── pinn_ode_inv_torch.py       # Inverse PINN (ODE)
│   │   └── pinn_pde_torch.py           # Forward PINN (PDE)
│   └── pinn_jax/                       # JAX + marimo apps
│       ├── pinn_ode_jax.py             # Forward PINN (ODE)
│       ├── pinn_ode_inv_jax.py         # Inverse PINN (ODE)
│       └── pinn_pde_jax.py             # Forward PINN (PDE)
└── scripts/                            # Standalone Python scripts
    ├── pinn_torch/                     # PyTorch script implementations
    │   ├── PINN_ODE_torch.py
    │   ├── PINN_ODE_inv_torch.py
    │   ├── PINN_PDE_torch.py
    └── pinn_jax/                       # JAX script implementations
        ├── PINN_ODE_jax.py
        ├── PINN_ODE_inv_jax.py
        └── PINN_PDE_jax.py
```
