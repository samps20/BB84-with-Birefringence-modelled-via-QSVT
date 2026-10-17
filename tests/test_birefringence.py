import numpy as np
from scipy.linalg import expm
from bb84_qsvt.birefringence import (
    compute_evolution_matrix,
    compute_qber,
    sift,
    simulate_transmission,
)

HBAR = 1.054571817e-34
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
L_TEST = 4e3
N_TEST = 40


def exact_step(L=L_TEST, N=N_TEST, v=2e8):
    dw, Om = 1e6, 5e5
    t_step = (L / v) / N
    return expm(-0.5j * (dw * Z + Om * X) * t_step)


def test_block_matches_exact_exponential():
    matrix, _ = compute_evolution_matrix(L=L_TEST, N=N_TEST)
    assert np.allclose(np.array(matrix), exact_step(L_TEST, N_TEST), atol=1e-3)


def test_n_steps_match_total_evolution():
    matrix, _ = compute_evolution_matrix(L=L_TEST, N=N_TEST)
    total = np.linalg.matrix_power(np.array(matrix), N_TEST)
    exact_total = exact_step(L_TEST, 1)  # N=1: all the evolution at once, no steps
    assert np.allclose(total, exact_total, atol=1e-2)


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
