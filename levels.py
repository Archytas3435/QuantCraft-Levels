from qiskit import QuantumCircuit, Aer, assemble
from math import sqrt

def get_vals(pct):
    return [sqrt(1-pct/100), sqrt(pct/100)]

def level1():
    num_quantum_registers = 1
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(50), 0)
    qc.measure(0, 0)
    qobj = assemble(qc)
    sim = Aer.get_backend("aer_simulator")
    counts = sim.run(qobj).result().get_counts()
    return counts
