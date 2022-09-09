from fractions import Fraction

import numpy as np
import sympy


class MarkovChainProblemSolver:
    def __init__(self, transition_matrix: list[list[tuple, int]]) -> None:
        if not isinstance(transition_matrix[0], list):
            raise TypeError("dimension of the transition matrix must be 2")
        if len(transition_matrix) != len(transition_matrix[0]):
            raise ValueError("number of rows and columns should be the same")
        self.P = np.array(
            [
                [Fraction(*i) if isinstance(i, tuple) else i for i in r]
                for r in transition_matrix
            ]
        )

    def transition_power(self, n: int) -> np.ndarray:
        """Probabilities from each row state to each column state after `n` steps

        Args:
            n (int): power of the transition matrix

        Returns:
            np.ndarray: probability matrix, P^n
        """
        F = self.P.copy()
        for _ in range(n - 1):
            F = F @ self.P
        return F

    def conditional_prob(
        self, init_state: int, occur: list[int, int], given: list[int, int]
    ) -> Fraction:
        """Probability that it is in `occur[1]` state when at the `occur[0]`th step
            given that it is in `given[1]` state when at the `given[0]`th step with
            the initial state (at 0th step) in `init_state`

        Args:
            init_state (int): initial state
            occur (list[int, int]): occur step and occur state
            given (list[int, int]): given step and given state

        Returns:
            Fraction: conditional probability
        """
        o_step, o_state = occur
        g_step, g_state = given
        if o_step == g_step:
            return Fraction(1)
        if o_step > g_step:
            return self.transition_power(o_step - g_step)[g_state, o_state]
        if o_step == 0:
            return Fraction(1) if o_state == init_state else Fraction(0)
        tp1 = self.transition_power(o_step)
        tp2 = self.transition_power(g_step - o_step)
        num = tp1[init_state, o_state] * tp2[o_state, g_state]
        dem = 0
        for i in range(len(self.P)):
            dem += tp1[init_state, i] * tp2[i, g_state]
        return num / dem

    def invariant_prob(self, state: int | None = None) -> np.ndarray | Fraction:
        """Long range fraction of time spent in each state

        Args:
            state (int | None, optional): state. Defaults to None.

        Returns:
            np.ndarray | Fraction: invariant probability or time fraction
        """
        A = sympy.Matrix(np.eye(len(self.P), dtype=Fraction) - self.P.T)
        b = sympy.Matrix([Fraction(0) for _ in range(len(self.P))])
        solution = np.array(next(iter(sympy.linsolve((A, b)))))
        solution /= solution.sum()
        solution = np.array([Fraction(i) for i in solution])
        return solution if state is None else solution[state]

    def mean_return_time(self, start_state: int) -> Fraction:
        """Expectated steps it takes to return to `start_state` from `start_state`

        Args:
            start_state (int): state where it starts

        Returns:
            Fraction: mean return time
        """
        return 1 / self.invariant_prob(start_state)

    def mean_passage_time(self, start_state: int, end_state: int) -> Fraction:
        """Expectated steps it takes to first visit `end_state` from `start_state`

        Args:
            start_state (int): state where it starts
            end_state (int): state where it ends

        Returns:
            Fraction: mean passage time
        """
        if start_state == end_state:
            raise ValueError("start state cannot be the same as end state")
        Q = np.delete(np.delete(self.P, start_state, 0), start_state, 1)
        M = np.array(sympy.Matrix(np.eye(len(Q), dtype=Fraction) - Q).inv())
        state = end_state if end_state < start_state else end_state - 1
        return Fraction(M[state, :].sum())


if __name__ == "__main__":
    P = [
        [(1, 8), (3, 8), (3, 8), (1, 8)],
        [(1, 4), (1, 2), (1, 4), 0],
        [(1, 2), (1, 2), 0, 0],
        [1, 0, 0, 0],
    ]

    ps = MarkovChainProblemSolver(P)
    # print(ps.conditional_prob(0, [3, 2], [4, 1]))
    # print(ps.transition_power(4))
    # print(ps.invariant_prob())
    # print(ps.mean_return_time(0))
    # print(ps.mean_passage_time(1, 3))
