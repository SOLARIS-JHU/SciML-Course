# EN.560.652 Scientific Machine Learning for Modeling, Optimization, and Control of Dynamical Systems  

Civil and Systems Engineering Department, Johns Hopkins University

> <p align="center">
>   <samp>
>     Nothing rests; everything moves; everything vibrates.
>   </samp>
> </p>
> <p align="right">
>   <samp>— <strong>Three Initiates</strong>, <em>Kybalion </em></samp>
> </p>

## Overview

This repository contains weekly lecture material and code examples for the SciML course.

- Week 1: Introduction 
- Week 2: Differentiable programming 
- Week 3: Physics-Informed Neural Networks 
- Week 4: Neural ODEs 
- Week 5: Learning to Optimize
- Week 6: Optimal control and MPC-style examples
- Week 7: Learning to Control 
- Week 8: Neural Differential Equations
- Week 9: Differentiable optimization 
- Week 10: Feasibility layers 

## Environment Installation

### Option 1: Conda

```bash
conda create -n sciml-course python=3.13 -y
conda activate sciml-course
python -m pip install -r requirements.txt
```

### Option 2: venv

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Option 3: Active environment

```bash
python -m pip install -r requirements.txt
```
