# Physics-Based CSTR Modeling and Machine Learning Surrogate

A study and extension project exploring **physics-based chemical process simulation, machine-learning surrogate modeling, and process optimization** using a Continuous Stirred-Tank Reactor (CSTR).

The project starts from an existing open-source **CSTR Hybrid Neural Network** implementation and aims to reproduce, understand, and extend the workflow by building a **Random Forest surrogate model** for the physics-based CSTR simulator.

---

## 1. Project Overview

A **Continuous Stirred-Tank Reactor (CSTR)** is a fundamental chemical-process system in which reactants continuously enter a well-mixed reactor and products continuously leave it.

The reactor behaviour can be described using physical laws such as:

- Mass balances
- Energy balances
- Reaction kinetics
- Arrhenius kinetics
- Ordinary Differential Equations (ODEs)

In this project, the CSTR acts as a **virtual chemical process**.

The physics-based simulator generates process data, and a machine-learning model learns to approximate the behaviour of the simulator.

### Overall Workflow

```text
                  CSTR PROCESS
                       |
                       v
              Physics-Based Model
                       |
              +--------+--------+
              |                 |
        Mass Balance       Energy Balance
              |                 |
              +--------+--------+
                       |
                       v
                  ODE Solver
                       |
                       v
              CSTR Simulation
                       |
                       v
             Generate Process Data
                       |
                       v
              Machine Learning
                       |
                       v
             Random Forest Model
                       |
                       v
              ML Surrogate Model
                       |
              +--------+--------+
              |                 |
       Physics Prediction   RF Prediction
              |                 |
              +--------+--------+
                       |
                       v
                Model Evaluation
                       |
          +------------+------------+
          |            |            |
         MAE          RMSE          R2
          |            |            |
          +------------+------------+
                       |
                       v
              Scientific Analysis
