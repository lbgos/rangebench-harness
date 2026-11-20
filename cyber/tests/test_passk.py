import math
import unittest

from rangebench.cli import pass_at_k


class PassAtKTests(unittest.TestCase):
    def test_no_solves_is_zero(self) -> None:
        self.assertEqual(pass_at_k(3, 0, 1), 0.0)

    def test_k_equals_n_with_one_solve_is_one(self) -> None:
        self.assertEqual(pass_at_k(3, 1, 3), 1.0)

    def test_pass_at_two_of_three(self) -> None:
        # c=2: every pair of trials contains a solve, 1 - C(1, 2) / C(3, 2) = 1.
        self.assertEqual(pass_at_k(3, 2, 2), 1.0)
        self.assertAlmostEqual(pass_at_k(3, 1, 2), 2 / 3)

    def test_one_of_five_at_three(self) -> None:
        self.assertAlmostEqual(pass_at_k(5, 1, 3), 1 - math.comb(4, 3) / math.comb(5, 3))
        self.assertAlmostEqual(pass_at_k(5, 1, 3), 0.6)

    def test_k_out_of_range_is_zero(self) -> None:
        self.assertEqual(pass_at_k(2, 2, 3), 0.0)
        self.assertEqual(pass_at_k(2, 2, 0), 0.0)


if __name__ == "__main__":
    unittest.main()
