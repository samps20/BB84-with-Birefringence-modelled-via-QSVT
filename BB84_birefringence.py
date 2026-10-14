import pennylane as qp
from pennylane import numpy as np

number_of_bits = 500

dev = qp.device("default.qubit", wires=4)

"""
Birefringence noise
"""


# define the cosine and sin angles that will be used for the QSVT
def cosine_angles(t):
    poly_cos = [
        1,
        0,
        -(t**2) / 2,
        0,
        (t**4) / 24,
    ]  # coefficientes of cosine taylor series
    angles_cos = qp.poly_to_angles(
        poly_cos, "QSVT"
    )  # obtain the angles for the QSVT to encode the cosine
    return angles_cos


def sine_angles(t):
    poly_sin = [0, t, 0, -(t**3) / 6, 0]  # coefficientes of sine taylor series
    angles_sin = qp.poly_to_angles(
        poly_sin, "QSVT"
    )  # obtain the angles for the QSVT to encode the sine
    return angles_sin


# hamiltonian for birefringence H=hbar/2 B*sigma (B is the birefringence matrix, sigma is the pauli matrices vector)
coeffs = np.array([5.273 * (10**-29), 2.636 * (10**-29)])
norm = np.sum(np.abs(coeffs))
coeffs_normalized = (coeffs / norm).tolist()

ops = [qp.PauliZ(3), qp.PauliX(3)]
hamiltonian = qp.dot(coeffs_normalized, ops)

# encoding of the hamiltonian
control_wires = [2]
block_encode = qp.PrepSelPrep(hamiltonian, control=control_wires)


# projector operators for the cosine and sine to create the polynomial transform with the just
# defined hamiltonian inside the block encoding
def cos_projectors(t):
    """
    Returns array of PCPhase gates with the parameters to implement
    approximation of cos(tx)
    """
    projectors = [qp.PCPhase(angle, dim=2, wires=[2, 3]) for angle in cosine_angles(t)]
    # dim indicates the dimension of the subspace inhabited by the Hamiltonian
    return projectors


def sin_projectors(t):
    projectors = [qp.PCPhase(angle, dim=2, wires=[2, 3]) for angle in sine_angles(t)]
    return projectors


# apply the whole QSVT protocol to apply the real part of the cosine(H_noramlized*t) and sin(H_normalized*t)
def circuit_cos(t):
    qp.Hadamard(1)
    qp.ctrl(qp.QSVT(block_encode, cos_projectors(t)), control=1, control_values=0)
    qp.ctrl(
        qp.adjoint(qp.QSVT(block_encode, cos_projectors(t))),
        control=1,
        control_values=1,
    )
    qp.Hadamard(1)


def circuit_sin(t):
    qp.Hadamard(1)
    qp.ctrl(qp.QSVT(block_encode, sin_projectors(t)), control=1, control_values=0)
    qp.ctrl(
        qp.adjoint(qp.QSVT(block_encode, sin_projectors(t))),
        control=1,
        control_values=1,
    )
    qp.Hadamard(1)


# Finally implement the whole circuit as cos(tx)+i*sin(tx)=e^(itx)
@qp.qnode(dev)
def qsvt_hamiltonian_evolution(t):
    qp.Hadamard(0)
    qp.ctrl(circuit_cos, control=0, control_values=0)(t)
    qp.S(0)
    qp.ctrl(circuit_sin, control=0, control_values=1)(t)
    qp.Hadamard(0)
    return qp.state()


work_wires = [3]
hbar = 1.054571817e-34
L, v = 1e3, 2e8  # 10 km of fiber, ~2e8 m/s
t_phys = L / v
t_eff = norm * t_phys / hbar  # ≈ 37 rad
N = 10  # step number, for Taylor expansion to be useful
t_step = t_eff / N
matrix = 2 * qp.matrix(
    qsvt_hamiltonian_evolution, wire_order=[0, 1] + control_wires + work_wires
)(
    t_step
)  # rescalated time

# finally, the matrix that represents the evolution operator
# slicing to obtain the just the matrix representing e^(i*H*t/hbar)
matrix_evolution = np.round(matrix[: 2 ** len(work_wires), : 2 ** len(work_wires)], 4)


def transmission(noise_matrix, steps):
    @qp.qnode(dev, shots=1)
    def one_transmission(alice_base, bit, bob_base, noise=noise_matrix, steps=steps):
        # Alice
        if alice_base == "Z" and bit == 1:
            qp.PauliX(3)
        elif alice_base == "X" and bit == 0:
            qp.Hadamard(3)
        elif alice_base == "X" and bit == 1:
            qp.PauliX(3)
            qp.Hadamard(3)

        # noise
        for h in range(steps):
            qp.QubitUnitary(noise, wires=3)

        # Bob
        if bob_base == "X":
            qp.Hadamard(3)

        return qp.sample(wires=3)

    # Alice random selection of bits and base
    alice_base_number = np.random.randint(0, 2)
    if alice_base_number == 0:
        base = "Z"
    elif alice_base_number == 1:
        base = "X"
    alice_bit = np.random.randint(0, 2)

    # Bob random selection of base
    bob_base_number = np.random.randint(0, 2)
    if bob_base_number == 1:
        bob_meas_base = "X"
    else:
        bob_meas_base = "Z"

    result = one_transmission(base, alice_bit, bob_meas_base)

    return (base, alice_bit, bob_meas_base, result[0][0])


alice_bits = []
alice_base = []
bob_bits = []
bob_base = []

for i in range(number_of_bits):
    a_base, a_bit, b_base, b_bit = transmission(matrix_evolution, N)
    alice_bits.append(a_bit)
    alice_base.append(a_base)
    bob_bits.append(int(b_bit))
    bob_base.append(b_base)

# print("Alice random bits:\t", alice_bits)
# print("Alice random bases:\t", alice_base)
# print("Bob random bits:\t", bob_bits)
# print("Bob random bases:\t", bob_base)

"""
Sifting
"""
save_position = []
for t in range(number_of_bits):
    if alice_base[t] == bob_base[t]:
        save_position.append(t)

# print("The common bases are in position:\t", save_position)

key = []
for position in save_position:
    key.append(alice_bits[position])

print("The secret key is:\t", key)

alice_key = [alice_bits[w] for w in save_position]
bob_key = [bob_bits[b] for b in save_position]

# QBER for the protocol
errors = sum(1 for a, b in zip(alice_key, bob_key) if a != b)
qber = errors / len(alice_key) if len(alice_key) > 0 else 0
print("Estimated QBER:\t", qber)

"""
El hamiltoniano y el qubit que evoluciona van a tener que ser el 3
el hamiltoniano es dimension 2x2 (pq solo actua sobre un qubit asi que dim 2^1)
los control wires que necesita es para solo 2 componentes (el de sigma_z y sigma_x) asi que log_2(2) redondeado hacia arriba es 1
Despues del block encoding tu hamiltoniano que actua en el 3 se mete dentro del block encoding que usa los wires [2,3] 
(para crearlo has usado el auxiliar y despues sobre el que actua).
Ahora creo los operadores de proyeccion que se encargaran de codificar el seno y el coseno y completar
la transformacion polinomica (de tus aputnes, el block encoding de H es U(A) y estos proyectores son Pi_phi, uno para cada angulo).
Ahora si que aplico el QSVT (usando la formula que tienes en los apuntes de intercalar Pi_phi con U(A)) para obtener
la parte real del coseno y del seno y ya tenerlos preparados.
Ahora, como pone en tus aputnes, para aplicar ese 1/2 e^(i*H_noramlied*t) es lo mismo que para sacar la parte real, pero usando lo del seno y el coseno.
Finalmente queremos sacar la matriz que representa ese e^(i*H_normalized*t) para ello primero sacamos
la matriz TOTAL (el block encoding y todo) y la multiplicamos por 2 para quitarnos ese 1/2 pesado de 1/2 e^(i*H_noramlied*t)
y de hecho, fijate arriba si reescalamos el tiempo por norm/hbar pasa esto en el exponente(i*(H/norm)norm*t/hbar)=i*H*t/bar sin normalizar
Por ultimo, de la matriz que te queda haces slicing, osea, te quedas solo un cacho, el que va de filas desde 0 hasta 2**len(work_wires)
y de columna 0 a 2**len(work wires), osea, el que contiene el exponente del hamiltoniano e^(i*H*t/hbar) (y bueno, lo redondea a 4 decimales)
sinceramente, voy a aplicar la evolucion un cojon de veces y ya. Es que sino t=37 y taylor explota, con t_steps=0.37 y aplicando 100 veces
la evolucoin es lo mismo (de forma exacta exp(t)=exp(t/N)^N asi que no hay problema)
"""
