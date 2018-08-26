from unittest import TestCase

import numpy as np

from usml_test_4 import *


class TestSolver(TestCase):

    def test_delete_bridge(self):
        matrix_distance = np.array([[0, 2, 0],
                                    [2, 0, 2],
                                    [0, 2, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        answer = solver.answer
        self.assertEqual(answer, 8)

    def test_delete_bridge_2(self):
        matrix_distance = np.array([[0, 2, 0, 2],
                                    [2, 0, 2, 2],
                                    [0, 2, 0, 0],
                                    [2, 2, 0, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        answer = solver.answer
        self.assertEqual(answer, 4)

    def test_solve_1(self):
        matrix_distance = np.array([[0, 2, 0, 2],
                                    [2, 0, 2, 2],
                                    [0, 2, 0, 0],
                                    [2, 2, 0, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertEqual(solver.solve(), 10)

    def test_solve_2(self):
        matrix_distance = np.array([[0, 1, 1, 1],
                                    [1, 0, 1, 1],
                                    [1, 1, 0, 1],
                                    [1, 1, 1, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertEqual(solver.solve(), 4)

    def test_solve_3(self):
        matrix_distance = np.array([[0, 1, 0, 1, 2],
                                    [1, 0, 1, 0, 0],
                                    [0, 1, 0, 1, 0],
                                    [1, 0, 1, 0, 10],
                                    [2, 0, 0, 10, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(7 <= solver.solve() < 11)

    def test_solve_4(self):
        matrix_distance = np.array([[0, 5, 1, 1, 5],
                                    [5, 0, 5, 1, 1],
                                    [1, 5, 0, 5, 1],
                                    [1, 1, 5, 0, 5],
                                    [5, 1, 1, 5, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(4 <= solver.solve() <= 8)

    def test_solve_5(self):
        matrix_distance = np.array([[0, 3, 2, 2, 4],
                                    [3, 0, 3, 2, 2],
                                    [2, 3, 0, 3, 2],
                                    [2, 2, 3, 0, 3],
                                    [4, 2, 2, 3, 0]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(9 <= solver.solve() <= 13)

    def test_solve_6(self):
        matrix_distance = np.array([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [1, 0, 2, 0, 0, 0, 7, 0, 0, 2],
                                    [0, 2, 0, 2, 0, 7, 0, 0, 8, 0],
                                    [0, 0, 2, 0, 1, 0, 7, 0, 0, 0],
                                    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0],
                                    [0, 0, 7, 0, 1, 0, 2, 0, 0, 0],
                                    [0, 7, 0, 7, 0, 2, 0, 1, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 1, 0, 1, 9],
                                    [0, 0, 8, 0, 0, 0, 0, 1, 0, 1],
                                    [0, 2, 0, 0, 0, 0, 0, 9, 1, 0]])

        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(14 <= solver.solve() <= 18)

    def test_solve_7(self):
        matrix_distance = np.array([[0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
                                    [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 1., 0., 1., 1., 1., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 1., 0., 0., 0., 0., 0., 0., 2.],
                                    [0., 0., 1., 0., 0., 0., 3., 0., 3., 0.],
                                    [0., 0., 0., 0., 0., 3., 0., 5., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 5., 0., 3., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 3., 0., 7.],
                                    [0., 0., 0., 0., 2., 0., 0., 0., 7., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(27 <= solver.solve() <= 31)

    def test_solve_8(self):
        matrix_distance = np.array([[0., 5., 0., 1., 2., 7.],
                                    [5., 0., 1., 0., 0., 1.],
                                    [0., 1., 0., 2., 0., 4.],
                                    [1., 0., 2., 0., 5., 0.],
                                    [2., 0., 0., 5., 0., 2.],
                                    [7., 1., 4., 0., 2., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(8 <= solver.solve() <= 12)

    def test_solve_9(self):
        matrix_distance = np.array([[0., 1., 3., 0., 0., 0., 0., 2., 0.],
                                    [1., 0., 2., 0., 0., 0., 0., 0., 0.],
                                    [3., 2., 0., 5., 0., 0., 0., 0., 0.],
                                    [0., 0., 5., 0., 0., 0., 0., 3., 0.],
                                    [0., 0., 0., 0., 0., 2., 3., 3., 0.],
                                    [0., 0., 0., 0., 2., 0., 1., 2., 0.],
                                    [0., 0., 0., 0., 3., 1., 0., 0., 3.],
                                    [2., 0., 0., 3., 3., 2., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 3., 0., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(27 <= solver.solve() <= 31)

    def test_solve_10(self):
        matrix_distance = np.array([[0., 0., 0., 0., 0., 3., 0., 0.],
                                    [0., 0., 1., 0., 0., 2., 0., 0.],
                                    [0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 1., 7., 0., 0.],
                                    [0., 0., 0., 1., 0., 2., 0., 0.],
                                    [3., 2., 0., 7., 2., 0., 2., 3.],
                                    [0., 0., 0., 0., 0., 2., 0., 0.],
                                    [0., 0., 0., 0., 0., 3., 0., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(27 <= solver.solve() <= 32)

    def test_solve_11(self):
        matrix_distance = np.array([[0., 3., 2., 2., 0., 0., 0., 1., 0.],
                                    [3., 0., 2., 0., 0., 1., 2., 2., 3.],
                                    [2., 2., 0., 1., 3., 2., 0., 0., 0.],
                                    [2., 0., 1., 0., 4., 0., 0., 0., 0.],
                                    [0., 0., 3., 4., 0., 2., 0., 0., 0.],
                                    [0., 1., 2., 0., 2., 0., 1., 0., 0.],
                                    [0., 2., 0., 0., 0., 1., 0., 0., 2.],
                                    [1., 2., 0., 0., 0., 0., 0., 0., 4.],
                                    [0., 3., 0., 0., 0., 0., 2., 4., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(13 <= solver.solve() <= 17)

    def test_solve_12(self):
        matrix_distance = np.array([[0., 1., 1., 1.],
                                    [1., 0., 1., 1.],
                                    [1., 1., 0., 1.],
                                    [1., 1., 1., 0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(3 <= solver.solve() <= 7)

    def test_solve_13(self):
        matrix_distance = np.array([[0.,  1.,  0.,  1.,  2.],
                                    [1.,  0.,  1.,  0.,  0.],
                                    [0.,  1.,  0.,  1.,  0.],
                                    [1.,  0.,  1.,  0., 10.],
                                    [2.,  0.,  0., 10.,  0.]])
        vertexs = get_vertexs(matrix_distance)
        solver = Solver(vertexs, matrix_distance)
        self.assertTrue(7 <= solver.solve() <= 11)
