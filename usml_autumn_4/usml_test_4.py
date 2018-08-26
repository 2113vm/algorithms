import numpy as np


def get_pair(arr):
    return np.nonzero(arr)[0], arr[arr > 0]


def get_vertexs(matrix_distance):
    return {enum: Vertex(enum=enum,
                         ns_distance={
                             neig_enum: dist
                             for neig_enum, dist in zip(*get_pair(matrix_distance[enum]))
                         })
            for enum, v in enumerate(matrix_distance)
            }


class Vertex:

    def __init__(self, enum, ns_distance: dict):
        self.enum = enum
        self.ns_distance = ns_distance
        self.is_bridge = True if len(ns_distance) == 1 else False
        self.is_last = True if len(ns_distance) == 0 else False

    def delete_ns(self, enum):
        self.ns_distance.pop(enum)
        if len(self.ns_distance) == 1:
            self.is_bridge = True
        if len(self.ns_distance) == 0:
            self.is_last = True

    def __str__(self):
        return ' | '.join([f'{self.enum + 1} -> {ns + 1} = {self.ns_distance[ns]}'
                           for ns in self.ns_distance])


class Solver:

    def __init__(self, vertexs, matrix_distance):
        self.vertexs = vertexs
        self.count_v = len(vertexs)
        self.set_vertexs = set(range(self.count_v))
        self.answer = 0
        self.start_v = 0
        self.delete_bridges()
        self.matrix_distance = matrix_distance
        self.tree = {}
        self.min_cost = None
        self.path_solve = [self.start_v]

    def solve(self):
        if self.count_v == 1:
            return self.answer
        else:
            self.find_path_()
            print(self.min_cost)
            print(self.tree)
            print(self.path_solve)
            return self.min_cost

    def find_path_(self):
        self.tree[self.start_v] = {}
        self.pull_tree_(tree=self.tree[self.start_v],
                        current_count_v=1,
                        current_vertex=self.start_v,
                        current_cost=self.answer,
                        set_vertexs=self.set_vertexs,
                        path=self.path_solve)

    def pull_tree_(self, tree, current_count_v, current_vertex, current_cost, set_vertexs, path):
        if len(path) > self.count_v + 3:
            return
        if len(set_vertexs) == 0 and current_vertex == self.start_v:
            self.min_cost = current_cost if self.min_cost is None or self.min_cost > current_cost \
                else self.min_cost
            tree[self.min_cost] = self.min_cost
            self.path_solve = path
            return
        for v in self.vertexs[current_vertex].ns_distance:
            if v in set_vertexs or v == self.start_v:
                tree[v] = {}
                current_path = path[:] + [v]
                cost = self.vertexs[current_vertex].ns_distance[v]
                self.pull_tree_(tree=tree[v],
                                current_count_v=current_count_v + 1,
                                current_vertex=v,
                                current_cost=current_cost + cost,
                                set_vertexs=set_vertexs - {current_vertex},
                                path=current_path)
            elif len(set_vertexs) == 1 and current_vertex in set_vertexs:
                ns = list(self.vertexs[current_vertex].ns_distance.keys())
                next_v = min(ns, key=lambda n: path.index(n))
                tree[next_v] = {}
                current_path = path[:] + [next_v]
                cost = self.vertexs[current_vertex].ns_distance[next_v]
                self.pull_tree_(tree=tree[next_v],
                                current_count_v=current_count_v + 1,
                                current_vertex=next_v,
                                current_cost=current_cost + cost,
                                set_vertexs={next_v},
                                path=current_path)

    def delete_bridges(self):
        for enum in self.vertexs:
            v = self.vertexs[enum]
            # if v.enum == 0:
            #     continue
            if v.is_bridge and not v.is_last:
                n = list(v.ns_distance.keys())[0]
                self.vertexs[n].delete_ns(v.enum)
                self.start_v = n if v.enum == self.start_v else self.start_v
                self.answer += 2 * v.ns_distance.pop(list(v.ns_distance.keys())[0])
                self.vertexs.pop(enum)
                self.count_v -= 1
                self.set_vertexs.remove(v.enum)
                self.delete_bridges()
                break

    def __str__(self):
        for v in self.vertexs:
            print(self.vertexs[v])
        return str(self.answer)


if __name__ == '__main__':

    N = int(input())

    matrix_distance = np.zeros((N, N))

    for v_i in range(N):
        V = list(map(int, input().split()))
        for v_j, distance in enumerate(V):
            matrix_distance[v_i, v_j] = distance

    vertexs = get_vertexs(matrix_distance)

    solver = Solver(vertexs=vertexs, matrix_distance=matrix_distance)
    print(solver)
