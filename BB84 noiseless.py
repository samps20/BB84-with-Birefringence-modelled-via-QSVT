import pennylane as qp
from pennylane import numpy as np

number_of_bits = 6

dev = qp.device("default.qubit", wires=1)


def transmission():
    @qp.qnode(dev, shots=1)
    def one_transmission(alice_base, bit, bob_base):
        # Alice
        if alice_base == "Z" and bit == 1:
            qp.PauliX(0)
        elif alice_base == "X" and bit == 0:
            qp.Hadamard(0)
        elif alice_base == "X" and bit == 1:
            qp.PauliX(0)
            qp.Hadamard(0)

        # Bob
        if bob_base == "X":
            qp.Hadamard(0)

        return qp.sample(wires=0)

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
    a_base, a_bit, b_base, b_bit = transmission()
    alice_bits.append(a_bit)
    alice_base.append(a_base)
    bob_bits.append(int(b_bit))
    bob_base.append(b_base)

print("Alice random bits:\t", alice_bits)
print("Alice random bases:\t", alice_base)
print("Bob random bits:\t", bob_bits)
print("Bob random bases:\t", bob_base)

"""
Sifting
"""
save_position = []
for t in range(number_of_bits):
    if alice_base[t] == bob_base[t]:
        save_position.append(t)

print("The common bases are in position:\t", save_position)

key = []
for position in save_position:
    key.append(alice_bits[position])

print("The secret key is:\t", key)
