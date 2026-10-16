import numpy as np

from bb84_qsvt.birefringence import (
    compute_evolution_matrix,
    compute_qber,
    sift,
    simulate_transmission,
)


def test_qber_with_hand_made_data():
    alice_key = [0, 1, 1, 0]
    bob_key = [0, 1, 0, 0]
    assert compute_qber(alice_key, bob_key) == 0.25


def test_qber_with_empty_key():
    assert compute_qber([], []) == 0


def test_sifting_with_hand_made_data():
    alice_bits = [0, 1, 1, 0]
    alice_base = ["Z", "X", "Z", "X"]
    bob_bits = [0, 0, 1, 1]
    bob_base = ["Z", "Z", "Z", "X"]

    positions, alice_key, bob_key = sift(alice_bits, alice_base, bob_bits, bob_base)

    assert positions == [0, 2, 3]
    assert alice_key == [0, 1, 0]
    assert bob_key == [0, 1, 1]


def test_evolution_matrix_is_unitary():
    # The QSVT approximation of e^(-iHt) must be (almost) unitary
    matrix, _ = compute_evolution_matrix()
    matrix = np.array(matrix)
    identity = matrix @ matrix.conj().T
    assert np.allclose(identity, np.eye(2), atol=1e-2)


def test_identity_noise_gives_zero_qber():
    # If the fiber does nothing, Alice and Bob must agree on every sifted bit
    a_bits, a_base, b_bits, b_base = simulate_transmission(30, np.eye(2), 1)
    _, alice_key, bob_key = sift(a_bits, a_base, b_bits, b_base)
    assert compute_qber(alice_key, bob_key) == 0
