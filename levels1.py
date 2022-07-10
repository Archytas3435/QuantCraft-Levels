from qiskit import QuantumCircuit, Aer, assemble, transpile
from math import sqrt

def get_vals(pct):
    return [sqrt(1-pct/100), sqrt(pct/100)]

def process(qc):
    sim = Aer.get_backend("aer_simulator")
    qc = transpile(qc, sim)
    qobj = assemble(qc)
    counts = sim.run(qobj, shots=100000).result().get_counts()
    return counts

def level1():
    num_quantum_registers = 1
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(50), 0)
    qc.measure(0, 0)
    return process(qc)

def level2():
    num_quantum_registers = 1
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(100), 0)
    qc.h(0)
    qc.measure(0, 0)
    return process(qc)

def level3():
    num_quantum_registers = 1
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(0), 0)
    qc.x(0)
    qc.measure(0, 0)
    return process(qc)

def level4():
    num_quantum_registers = 2
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(100), 0)
    qc.initialize(get_vals(0), 1)
    qc.cx(0, 1)
    qc.measure(1, 0)
    return process(qc)

def level5():
    num_quantum_registers = 1
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(50), 0)
    qc.h(0)
    qc.measure(0, 0)
    return process(qc)

def level6():
    num_quantum_registers = 2
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(0), 0)
    qc.initialize(get_vals(0), 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure(1, 0)
    return process(qc)

def level7():
    num_quantum_registers = 2
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(25), 0)
    qc.initialize(get_vals(75), 1)
    qc.cx(0, 1)
    qc.measure(1, 0)
    return process(qc)

def level8():
    num_quantum_registers = 2
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(50), 0)
    qc.initialize(get_vals(0), 1)
    qc.ch(0, 1)
    qc.measure(1, 0)
    return process(qc)

def level9():
    num_quantum_registers = 3
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(0), 0)
    qc.initialize(get_vals(0), 1)
    qc.initialize(get_vals(0), 2)
    qc.h(0)
    qc.ch(0, 1)
    qc.ch(1, 2)
    qc.measure(2, 0)
    return process(qc)

def level10():
    num_quantum_registers = 3
    num_classical_registers = 1
    qc = QuantumCircuit(num_quantum_registers, num_classical_registers)
    qc.initialize(get_vals(0), 0)
    qc.initialize(get_vals(0), 1)
    qc.initialize(get_vals(25), 2)
    qc.h(0)
    qc.ch(0, 1)
    qc.cx(1, 2)
    qc.measure(2, 0)
    return process(qc)
