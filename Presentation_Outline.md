# Presentation Outline: Quantum Route Optimization

## Slide 1: Title Slide
- **Title**: Quantum Route Optimization using QAOA
- **Subtitle**: Solving the Traveling Salesperson Problem (TSP) on IBM Quantum Hardware
- **Presenter**: [Your Name]
- **Date**: [Current Date]

## Slide 2: The Problem: Route Optimization (TSP)
- **What is TSP?**: Given a list of cities and the distances between each pair of cities, what is the shortest possible route that visits each city exactly once and returns to the origin city?
- **Real-World Applications**: Logistics, delivery routing, network design, DNA sequencing.
- **Why is it hard?**: It's an NP-hard problem. The number of possible routes grows factorially (N!) with the number of cities.
  - 4 cities = 3 unique routes
  - 10 cities = 181,440 routes
  - 20 cities = 60 quintillion routes!

## Slide 3: The Solution: Quantum Computing
- **Quantum Advantage**: Quantum computers can explore vast solution spaces more efficiently than classical computers for certain problems.
- **QAOA (Quantum Approximate Optimization Algorithm)**:
  - A hybrid quantum-classical algorithm designed for combinatorial optimization problems.
  - Uses a parameterized quantum circuit (Ansatz) to find the "lowest energy state" of a Hamiltonian, which corresponds to the optimal route.
  - Can run on current NISQ (Noisy Intermediate-Scale Quantum) devices.

## Slide 4: System Architecture
- **Input**: User selects the number of cities (nodes) and random seed.
- **Problem Formulation**:
  - Cities and distances -> Graph -> Quadratic Program (Ising Hamiltonian).
  - Encoded into Qubits (N^2 qubits required for N cities).
- **Quantum Execution**:
  - Circuit construction using Qiskit.
  - Execution on **IBM Quantum Hardware** (e.g., `ibm_brisbane`) or local simulator.
- **Optimization**: Classical optimizer (COBYLA/SPSA) tunes quantum parameters.
- **Output**: Optimized route displayed on a 2D map.

## Slide 5: Implementation Details
- **Tech Stack**:
  - **Frontend**: HTML/CSS/JS (Flask Templates)
  - **Backend**: Python (Flask)
  - **Quantum SDK**: Qiskit, Qiskit Optimization, Qiskit IBM Runtime
- **Key Modules**:
  - `QAOA_qiskit.py`: Handles quantum circuit generation and execution.
  - `utilities.py`: Generates TSP instances and provides classical brute-force benchmark.

## Slide 6: Results & Comparison
- **Visual Comparison**:
  - **Classical Route**: Solved via Brute Force (Ground Truth).
  - **Quantum Route**: Solved via QAOA.
- **Performance Metrics**:
  - **Cost (Distance)**: Comparison of total path length.
  - **Execution Time**: Time taken for classical vs. quantum execution.
  - **Accuracy**: How close the quantum solution is to the optimal classical solution.

## Slide 7: Challenges & Future Work
- **Hardware Limitations**:
  - Qubit Count: TSP is expensive (N^2 qubits). Current hardware limits us to small instances (4-5 cities).
  - Noise: Quantum noise affects accuracy on real hardware compared to simulators.
- **Future Improvements**:
  - Use more efficient encodings (e.g., Logarithmic encoding) to reduce qubit usage.
  - Implement error mitigation techniques.
  - Scale to larger problems as hardware matures.

## Slide 8: Conclusion
- Successfully demonstrated end-to-end execution of a hard optimization problem on real quantum hardware.
- The application provides an interactive interface for exploring quantum algorithms.
- Quantum computing shows promise for future large-scale logistics optimization.

## Slide 9: Q&A
- Thank you! Questions?
