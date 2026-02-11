# How the Project Works: Step-by-Step Guide

This document explains exactly how your Quantum Route Optimization (TSP) application works, distinguishing clearly between the **Classical** parts and the **Quantum** parts.

---

## 1. The User Interface (Frontend)
**File:** `templates/index.html`
- **What happens:** You see a dashboard where you select the **Number of Cities** (e.g., 4) and the **Quantum Backend** (Simulator or Real Hardware).
- **Action:** When you click **"Run Simulation"**, the webpage packages your choices into a bundle (JSON) and sends it to the server.

---

## 2. The Server Controller (Backend)
**File:** `flask_app/app.py`
- **Role:** This is the "Brain". It coordinates everything.
- **Step 1: Data Generation**
  - It calls `generate_tsp_instance(n_nodes)` from `utilities.py`.
  - **What it does:** It creates random X, Y coordinates for 4 cities and calculates the distance between every pair of cities.
  
- **Step 2: The "Classical" Benchmark (The Control Group)**
  - It calls `solve_tsp_brute_force()` from `utilities.py`.
  - **How it works:** This is a **purely classical** python function. It uses the `itertools` library to generate every possible route list (e.g., `[0,1,2,3]`, `[0,2,1,3]`, etc.), calculates the distance for each, and picks the smallest one.
  - **Purpose:** This gives us the "Correct Answer" to see if the quantum computer is right or wrong.

- **Step 3: The "Quantum" Solver (The Experiment)**
  - It calls `qaoa_tsp()` from `src/QAOA_qiskit.py`.
  - **This is where the Quantum Magic happens.** (Detailed below).

---

## 3. The Quantum Core
**File:** `src/QAOA_qiskit.py`
This entire file is dedicated to running the Quantum Algorithm. Here is the flow inside `qaoa_tsp`:

### A. Translation (Problem -> Math)
- The list of cities and distances is converted into a **"Quadratic Program"**.
- This is further converted into an **"Ising Hamiltonian"** (a giant matrix representing energy systems).
- **Physical Meaning:** In this system, "Low Energy" = "Short Distance". We want to find the lowest energy state.

### B. Circuit Construction (Math -> Circuit)
- The code uses **Qiskit** to build a **Quantum Circuit**.
- It creates **N² Qubits** (16 qubits for 4 cities).
- It applies a specific pattern of quantum gates (Hadamard gates, CNOTs, Rotations) defined by the **QAOA Ansatz**.

### C. Execution (Circuit -> Hardware/Simulator)
- **If Simulator:** The circuit runs on your local CPU, simulating how quantum probabilities would behave.
- **If Real Hardware:** The circuit is sent over the internet to IBM's lab. Real electrons/photons interact on a chip to run your circuit.

### D. Optimization & Measurement
- The algorithm runs in a loop (Hybrid Loop):
  1. Run Circuit.
  2. Measure output (get a bitstring like `10010010...`).
  3. Calculate Cost.
  4. Classical optimizer (COBYLA) tunes the gate angles.
  5. Repeat until the best result is found.

### E. Interpretation (Bitstring -> Route)
- The final binary string (e.g., `1000010000100001`) is decoded back into a city route (e.g., "City 0 -> City 2 -> City 1 -> City 3").

---

## 4. Visualization
**File:** `app.py` (helpers)
- The server takes the **Classical Path** and the **Quantum Path**.
- It draws two maps using **Plotly**.
- Note: Sometimes the Quantum path might be different (or even invalid/broken) compared to the Classical path. This is expected! Quantum computers are probabilistic and noisy.

---

## Summary Checklist

| Feature | Code File | Method | Meaning |
| :--- | :--- | :--- | :--- |
| **Input** | `index.html` | Form | User Config |
| **Logic** | `app.py` | `/run_simulation` | The Manager |
| **Data** | `utilities.py` | `generate_tsp_instance` | Random Map Creation |
| **Classical** | `utilities.py` | `solve_tsp_brute_force` | Traditional Math (100% accurate) |
| **Quantum** | `QAOA_qiskit.py` | `qaoa_tsp` | **The Actual Quantum Algorithm** |

