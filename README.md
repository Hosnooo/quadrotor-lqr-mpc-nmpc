# quadrotor-lqr-mpc-nmpc

Python simulation framework for comparing optimal control strategies on a Quanser QDrone2 quadrotor model.

This project implements and compares several controllers for position and yaw tracking near hover:

- Infinite-horizon Linear Quadratic Regulator (LQR)
- Linear Model Predictive Control (Linear MPC)
- Nonlinear Model Predictive Control (NMPC)
- Nonlinear optimal control using Pontryagin’s Minimum Principle (PMP) solved by direct multiple shooting with CasADi/Ipopt

The project uses a 12-state quadrotor rigid-body model in the NED frame with ZYX Euler angles. Rotor-speed limits are mapped consistently into thrust and torque constraints through the same allocation model.

---

## Project Overview

The goal of this project is to evaluate how different optimal control methods perform on the same quadrotor tracking task.

The tracking reference is a position-yaw step command:

```text
p_ref = [1, -1, -3] m
psi_ref = 0.05 rad
```

The compared controllers use the same model structure, cost weights, actuator limits, and simulation duration to allow a fair comparison.

---

## Controllers

### 1. LQR

An infinite-horizon LQR controller is designed using the linearized hover model. The controller is applied to both the linearized plant and the full nonlinear plant.

### 2. Linear MPC

The linear MPC controller uses the hover-linearized model in deviation form. The continuous-time linear model is discretized using zero-order hold, and the MPC problem is solved as a quadratic program.

### 3. Nonlinear MPC

The nonlinear MPC controller uses the full nonlinear quadrotor dynamics. The prediction model is discretized using RK4 integration, and the controller is formulated as a multiple-shooting nonlinear program.

### 4. PMP / Direct Multiple Shooting OCP

A finite-horizon nonlinear optimal control problem is solved using direct multiple shooting with CasADi and Ipopt. This gives a benchmark optimal trajectory and includes PMP residual diagnostics.

---

## Repository Structure

```text
.
├── Artifacts/
│   ├── lqr_hover_.npz
│   ├── lqr_hover__report.txt
│   └── LQR gain and Riccati matrix plots
│
├── Simulations/
│   ├── lqr_lin_*.npz / *.pdf
│   ├── lqr_nonlin_*.npz / *.pdf
│   ├── mpc_lin_*.npz / *.pdf
│   ├── mpc_nonlin_*.npz / *.pdf
│   ├── ocp_nonlin_ms_casadi_*.npz / *.pdf
│   ├── NMPC_sweepN_dt020/
│   └── NMPC_sweepConstraints_dt020_N080/
│
├── codes/
│   ├── dynamics.py
│   ├── lqr.py
│   ├── mpc_linear.py
│   ├── mpc_nonlinear.py
│   ├── casadi_ms.py
│   ├── sim_lin_lqr.py
│   ├── sim_nonlin_lqr.py
│   ├── sim_mpc_linear.py
│   ├── sim_mpc_nonlinear.py
│   ├── sim_nonlin_casadi_ms.py
│   ├── sim_nmpc_sweepN_runner.py
│   ├── sim_nmpc_sweepConstraints_runner.py
│   ├── plotting.py
│   ├── utils.py
│   └── macros.py
│
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/qdrone2-optimal-control.git
cd qdrone2-optimal-control
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

```bash
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Requirements

The main dependencies are:

```text
numpy
scipy
matplotlib
casadi
cvxpy
```

CasADi/Ipopt is used for the nonlinear MPC and direct multiple-shooting optimal control problems. CVXPY is used for the linear MPC quadratic program.

---

## Running the Simulations

Run the LQR design:

```bash
python codes/lqr.py
```

Run LQR on the linearized plant:

```bash
python codes/sim_lin_lqr.py
```

Run LQR on the nonlinear plant:

```bash
python codes/sim_nonlin_lqr.py
```

Run Linear MPC:

```bash
python codes/sim_mpc_linear.py
```

Run Nonlinear MPC:

```bash
python codes/sim_mpc_nonlinear.py
```

Run the nonlinear OCP using CasADi/Ipopt:

```bash
python codes/sim_nonlin_casadi_ms.py
```

Run the NMPC horizon sweep:

```bash
python codes/sim_nmpc_sweepN_runner.py
python codes/plot_nmpc_sweepN_summary.py
```

Run the NMPC actuator-limit sweep:

```bash
python codes/sim_nmpc_sweepConstraints_runner.py
python codes/plot_nmpc_sweepConstraints_summary.py
```

Simulation results are saved in the `Simulations/` folder as `.npz` data files and `.pdf` plots.

---

## Main Model

The quadrotor state is

```text
x = [p, v, eta, omega]
```

where:

```text
p     = [x, y, z]              position in NED frame
v     = [vx, vy, vz]           velocity in NED frame
eta   = [phi, theta, psi]      ZYX Euler angles
omega = [p, q, r]              body angular rates
```

The control input is

```text
u = [T, tau_x, tau_y, tau_z]
```

where `T` is the total thrust and `tau_x`, `tau_y`, `tau_z` are body torques.

The rotor allocation maps squared motor speeds to thrust and torque:

```text
[T, tau_x, tau_y, tau_z]^T = M [Omega_1^2, Omega_2^2, Omega_3^2, Omega_4^2]^T
```

---

## Simulation Settings

The default simulation uses:

```text
Simulation time: 8 s
Reference position: [1, -1, -3] m
Reference yaw: 0.05 rad
Maximum motor speed: 20000 RPM
Minimum motor speed: 0 RPM
```

Controller weights and physical constants are defined in:

```text
codes/macros.py
```

---

## Results Summary

The project compares the controllers using:

- Total quadratic objective value
- Input saturation count
- 2% settling time
- Overshoot
- Peak roll and pitch angles
- Rotor-speed behavior
- PMP residual diagnostics for the nonlinear OCP

In the reported comparison, the PMP/Ipopt direct multiple-shooting solution achieved the lowest objective value, while LQR and Linear MPC provided strong real-time baseline performance. The coarse-grid NMPC case showed higher cost and larger attitude/yaw peaks, indicating that NMPC performance is sensitive to sampling time and horizon length.

---

## Generated Outputs

Typical output files include:

```text
Simulations/lqr_lin__states.pdf
Simulations/lqr_lin__rpms.pdf
Simulations/lqr_nonlin__states.pdf
Simulations/lqr_nonlin__rpms.pdf
Simulations/mpc_lin__states.pdf
Simulations/mpc_lin__rpms.pdf
Simulations/mpc_nonlin__states.pdf
Simulations/mpc_nonlin__rpms.pdf
Simulations/ocp_nonlin_ms_casadi__states.pdf
Simulations/ocp_nonlin_ms_casadi__rpms.pdf
Simulations/ocp_nonlin_ms_casadi__costates.pdf
Simulations/ocp_nonlin_ms_casadi__pmp_residuals.pdf
```

---

## Notes

- The code uses the NED convention, where positive `z` points downward.
- The thrust direction is opposite the body positive `z` axis.
- The model uses ZYX Euler angles, so very large pitch angles should be avoided due to Euler-angle singularity.
- Rotor speed limits are enforced through the allocation and saturation functions.
- NMPC and OCP scripts may require more computation time than LQR and Linear MPC.

---

## Author

Mohssen E. Elshaar

---

## License

Add your preferred license here.

For example:

```text
MIT License
```
