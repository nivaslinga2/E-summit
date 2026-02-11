# Understanding Qubits in Route Optimization (TSP)

## 1. What is a "Qubit" in this context?

Think of a **Qubit** (Quantum Bit) as a **switch** that can be both **ON (1)** and **OFF (0)** at the same time (superposition) until we measure it.

In our Route Optimization problem, each qubit represents a specific **decision**:
**"Is City `A` visited at Time Step `T`?"**

If the qubit is **1**, the answer is **YES**.
If the qubit is **0**, the answer is **NO**.

---

## 2. Why do we need 16 Qubits for 4 Cities?

The Traveling Salesperson Problem (TSP) asks for an **ORDER** of visiting cities.
For **4 Cities** (e.g., City 0, City 1, City 2, City 3), we need to decide **which city is visited at each step of the journey**.

The journey has **4 Steps** (First stop, Second stop, Third stop, Fourth stop).

So, we need a grid of **Cities x Time Steps**:

| Time Step | City 0 | City 1 | City 2 | City 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Stop 1** | Qubit 0 | Qubit 1 | Qubit 2 | Qubit 3 |
| **Stop 2** | Qubit 4 | Qubit 5 | Qubit 6 | Qubit 7 |
| **Stop 3** | Qubit 8 | Qubit 9 | Qubit 10| Qubit 11|
| **Stop 4** | Qubit 12| Qubit 13| Qubit 14| Qubit 15|

**Total Qubits = Number of Cities (4) × Number of Steps (4) = 16 Qubits.**

---

## 3. How does the Quantum Computer solve this?

The quantum computer doesn't just check one route at a time. It uses **entanglement** and **superposition** to explore combinations of these 16 qubits simultaneously.

It looks for a pattern of 1s and 0s that satisfies two main rules (Constraints):
1.  **Each City is visited EXACTLY ONCE.** (Each Column must have exactly one '1')
2.  **Each Time Step has EXACTLY ONE City.** (Each Row must have exactly one '1')

**Example of a Valid Route (0 -> 2 -> 1 -> 3):**

| Time Step | City 0 | City 1 | City 2 | City 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Stop 1** | **1** | 0 | 0 | 0 |
| **Stop 2** | 0 | 0 | **1** | 0 |
| **Stop 3** | 0 | **1** | 0 | 0 |
| **Stop 4** | 0 | 0 | 0 | **1** |

The quantum algorithm (QAOA) tries to find this specific grid (bitstring) that gives the **shortest total distance**.

---

## Summary

- **Qubit**: Represents a "Yes/No" question: "Is City X visited at Step Y?"
- **16 Qubits**: Because we have a 4x4 grid of possibilities (4 Cities × 4 Time Steps).
- **Quantum Power**: Instead of checking every possible grid one by one (classical brute force), the quantum computer uses quantum physics to find the best grid configuration (lowest energy state).
