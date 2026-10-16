from bb84_qsvt.noiseless import simulate_transmission, sift


def test_sifting_with_hand_made_data():
    # No quantum simulation: only checks the sifting logic
    alice_bits = [0, 1, 1, 0]
    alice_base = ["Z", "X", "Z", "X"]
    bob_bits = [0, 0, 1, 1]
    bob_base = ["Z", "Z", "Z", "X"]

    positions, key = sift(alice_bits, alice_base, bob_bits, bob_base)

    assert positions == [0, 2, 3]
    assert key == [0, 1, 0]


def test_noiseless_keys_match():
    # Without noise or eavesdropper, matching bases give identical bits
    for _ in range(5):
        a_bits, a_base, b_bits, b_base = simulate_transmission(100)
        positions, key = sift(a_bits, a_base, b_bits, b_base)
        bob_key = [b_bits[i] for i in positions]
        assert key == bob_key


def test_sifting_keeps_about_half_of_the_bits():
    # Bases are random, so they should match roughly 50% of the time
    a_bits, a_base, b_bits, b_base = simulate_transmission(400)
    positions, _ = sift(a_bits, a_base, b_bits, b_base)
    assert 0.35 < len(positions) / 400 < 0.65
