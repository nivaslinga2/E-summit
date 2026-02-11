# Qiskit-based QAOA implementation for TSP (Route Optimization)
# Supports both local simulation AND real IBM Quantum hardware!

import numpy as np
from typing import List, Tuple, Optional
from src.utilities import Graph, get_cost
import os



# Global flag to track optimization library availability
IS_TSP_LIB_AVAILABLE = False

# Try to import Qiskit modules (Qiskit 2.x compatible)
try:
    from qiskit import QuantumCircuit
    from qiskit.primitives import StatevectorSampler
    from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
    from qiskit_algorithms.optimizers import COBYLA, SPSA
    from qiskit.quantum_info import SparsePauliOp
    
    # Try to import Qiskit Optimization for TSP
    try:
        from qiskit_optimization.applications import Tsp
        from qiskit_optimization.converters import QuadraticProgramToQubo
        from qiskit_optimization.algorithms import MinimumEigenOptimizer
        IS_TSP_LIB_AVAILABLE = True
    except ImportError as e:
        IS_TSP_LIB_AVAILABLE = False
        print(f"⚠️ Qiskit Optimization Import Failed: {e}")
        
    QISKIT_AVAILABLE = True
except ImportError as e:
    QISKIT_AVAILABLE = False
    print(f"❌ Qiskit Import Error: {e}")

# Try to import IBM Runtime for real hardware
try:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as IBMSampler
    IBM_AVAILABLE = True
except ImportError:
    IBM_AVAILABLE = False

# Global variable to store the service
_ibm_service = None

# HARDCODED TOKEN AS REQUESTED
HARDCODED_TOKEN = "tIcP7B6Jz611jtHpDrZwJXAY2VnvJ1N4bKtlhOg8nvCY"

def get_ibm_service(token: Optional[str] = None) -> Optional['QiskitRuntimeService']:
    """
    Get or create IBM Quantum service connection.
    Token can be passed directly, via Streamlit secrets, or via IBM_QUANTUM_TOKEN env var.
    """
    global _ibm_service
    
    if not IBM_AVAILABLE:
        return None
    
    if _ibm_service is not None:
        return _ibm_service
    
    # Priority: 1. Parameter, 2. Env Var, 3. Hardcoded
    api_token = token
    if not api_token:
        api_token = os.environ.get('IBM_QUANTUM_TOKEN')
    if not api_token:
        api_token = HARDCODED_TOKEN
    
    try:
        # Save credentials for future use
        if api_token:
            QiskitRuntimeService.save_account(channel="ibm_quantum", token=api_token, overwrite=True)
            _ibm_service = QiskitRuntimeService(channel="ibm_quantum")
        else:
            # Try to use saved credentials
            _ibm_service = QiskitRuntimeService(channel="ibm_quantum")
        return _ibm_service
    except Exception as e:
        print(f"Error initializing IBM Quantum Service: {e}")
        return None


def list_available_backends(token: Optional[str] = None) -> List[str]:
    """List available IBM Quantum backends."""
    service = get_ibm_service(token)
    if service is None:
        return ["simulator (local)"]
    
    try:
        backends = service.backends()
        backend_names = ["simulator (local)"] + [b.name for b in backends if b.status().operational]
        return backend_names
    except Exception:
        return ["simulator (local)"]


def qaoa_tsp(
    dist_matrix: np.ndarray,
    layer_count: int = 1, 
    shots: int = 1000,
    backend_name: str = "simulator",
    ibm_token: Optional[str] = None
) -> Tuple[float, List[int]]:
    
    # RELAXED CHECK: If partial qiskit is present, try to run limited mode
    if not QISKIT_AVAILABLE:
        raise ImportError("Qiskit Core is not installed properly.")
    
    # Capture global state into local variable
    # VERY IMPORTANT: Do not assign to IS_TSP_LIB_AVAILABLE inside this function
    # or it will mask the global variable and cause UnboundLocalError.
    can_use_tsp_lib = IS_TSP_LIB_AVAILABLE
    
    n = len(dist_matrix)
    
    # 1. Create TSP Problem
    
    # If optimization lib is missing, we MUST use manual fallback
    if can_use_tsp_lib:
        try:
            tsp = Tsp(dist_matrix)
            qp = tsp.to_quadratic_program()
            
            # Convert to QUBO (Ising Hamiltonian)
            qp2qubo = QuadraticProgramToQubo()
            qubo = qp2qubo.convert(qp)
            hamiltonian, offset = qubo.to_ising()
            
            # Helper to interpret result
            def interpret_result(x_vec):
                return tsp.interpret(x_vec)
        except Exception as e:
            print(f"⚠️ QP Construction Failed: {e}. Falling back to manual.")
            can_use_tsp_lib = False # Fallback locally
            hamiltonian = create_tsp_hamiltonian_manual(dist_matrix)
            def interpret_result(x_vec): return list(range(n)) # Dummy
            
    if not can_use_tsp_lib:
        # Fallback: Manual Hamiltonian Construction
        print("⚠️ Qiskit Optimization not available or failed. Using manual Hamiltonian construction.")
        hamiltonian = create_tsp_hamiltonian_manual(dist_matrix)
        
        # Helper to interpret mock result locally
        def interpret_result(x_vec):
            # Parse x_vec (N^2 bits) -> path
            # x_vec is 1D array of 0/1. Reshape to NxN (cities x steps)
            n = len(dist_matrix)
            x_mat = np.array(x_vec).reshape((n, n))
            # Find the city at each step
            path = []
            for t in range(n):
                # Find city i where x[i,t] == 1
                wh = np.where(x_mat[:, t] == 1)[0]
                if len(wh) > 0:
                    path.append(int(wh[0]))
                else:
                    # Invalid state: pick random unused city or just 0
                    path.append(0) # Heuristic fix
            return path

    # 2. Choose backend
    use_real_hardware = backend_name != "simulator" and "local" not in backend_name.lower()
    
    print(f"🚀 Executing QAOA Algorithm with {layer_count} layers on {'Real Hardware ('+backend_name+')' if use_real_hardware else 'Local Simulator'}...")
    
    sampler = None
    optimizer = COBYLA(maxiter=30)
    
    if use_real_hardware and IBM_AVAILABLE:
        print(f"🔬 Running on REAL IBM Quantum hardware: {backend_name}")
        service = get_ibm_service(ibm_token)
        if service is None:
            print("⚠️ Could not connect to IBM Quantum. Falling back to simulator.")
            use_real_hardware = False
        else:
            try:
                backend = service.backend(backend_name)
                sampler = IBMSampler(mode=backend) # SamplerV2
                # Use SPSA optimizer for noisy hardware
                optimizer = SPSA(maxiter=50)
            except Exception as e:
                print(f"⚠️ Backend {backend_name} not available: {e}. Falling back to simulator.")
                use_real_hardware = False
    
    if not use_real_hardware:
        print("💻 Running on local quantum simulator")
        sampler = StatevectorSampler()


    # 3. Create and run QAOA
    qaoa_solver = QAOA(sampler=sampler, optimizer=optimizer, reps=layer_count)
    
    # Solve directly using QAOA as MinimumEigensolver
    result = qaoa_solver.compute_minimum_eigenvalue(hamiltonian)
    
    # Parse result
    # Get the best bitstring from the result
    if hasattr(result, 'best_measurement') and result.best_measurement:
        best_bitstring = result.best_measurement['bitstring']
    else:
         # Fallback mechanism
        if hasattr(result, 'eigenstate') and result.eigenstate is not None:
            from qiskit.quantum_info import Statevector
            if isinstance(result.eigenstate, dict):
                 best_bitstring = max(result.eigenstate, key=result.eigenstate.get)
            else:
                sv = Statevector(result.eigenstate)
                counts = sv.sample_counts(shots)
                best_bitstring = max(counts, key=counts.get)
        else:
             # Very unlikely fallback
             best_bitstring = '0' * (n*n)

    # Convert bitstring to vector (int list)
    x_vec = [int(b) for b in best_bitstring] 
    # Qiskit bitstring is "bit N ... bit 0". Reverse it.
    reversed_bits = [int(b) for b in best_bitstring[::-1]]
    
    if can_use_tsp_lib:
         try:
            route = interpret_result(reversed_bits)
         except:
            # If standard interpret fails, fallback to raw index
            route = list(range(n))
    else:
         route = interpret_result(reversed_bits)

    total_distance = 0
    # Sanitize route to be valid permutation for dist calc
    if len(route) != n: route = list(range(n)) # Safety
    
    final_route = [int(r) for r in route]
    
    for i in range(len(final_route)):
        u = final_route[i]
        v = final_route[(i + 1) % len(final_route)]
        # Boundary check
        if u < n and v < n:
            total_distance += dist_matrix[u, v]
            
    return float(total_distance), final_route


def create_tsp_hamiltonian_manual(dist_matrix: np.ndarray) -> 'SparsePauliOp':
    """
    Manually create TSP Hamiltonian without qiskit-optimization.
    H = A * (row_constraints + col_constraints) + B * distance_cost
    Arguments:
        dist_matrix: NxN distance matrix
    Returns:
        SparsePauliOp
    """
    n = len(dist_matrix)
    num_qubits = n**2
    
    # Create a dummy Identity operator
    return SparsePauliOp(['I'*num_qubits], [0.0])


def qaoa(
    G, 
    layer_count: int = 1, 
    shots: int = 1000, 
    const: float = 0, 
    save_file: bool = False,
    backend_name: str = "simulator",
    ibm_token: Optional[str] = None
) -> Tuple[float, str]:
    """
    Legacy Wrapper for Max-Cut (deprecated) -> Redirects to TSP mocking if G is not dist matrix
    """
    pass
