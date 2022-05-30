import numpy as np
from scipy.optimize import lsq_linear
from sortedcontainers import SortedList
from timeit import default_timer
from tqdm import tqdm
from copy import deepcopy


def solver(H, y, bounds=None):
    solution = lsq_linear(H, y, bounds=bounds, max_iter=100)
    return solution.x, ((H @ solution.x - y) ** 2).sum()


def strong_branching(node):
    best_i = None
    best_score = None
    for i in range(len(node['x_relaxed'])):
        if np.isclose(node['x_relaxed'][i], node['x_rounded'][i]):
            continue
        lb, ub = node['lb'].copy(), node['ub'].copy()
        H, y = node['H'], node['y']
        ub_i = np.floor(node['x_relaxed'][i])
        ub[i] = ub_i
        if np.isclose(ub[i], lb[i]):
            y = node['y'] - node['H'][:, i] * ub_i
            H = np.concatenate([node['H'][:, :i], node['H'][:, i+1:]], axis=1)
            ub.pop(i)
            lb.pop(i)
        x_relaxed, lower_bound = solver(H, y, bounds=(lb, ub))
        score_left = lower_bound - node['lower_bound']

        lb, ub = node['lb'].copy(), node['ub'].copy()
        H, y = node['H'], node['y']
        lb_i = np.ceil(node['x_relaxed'][i])
        lb[i] = lb_i
        if np.isclose(ub[i], lb[i]):
            y = node['y'] - node['H'][:, i] * lb_i
            H = np.concatenate([node['H'][:, :i], node['H'][:, i+1:]], axis=1)
            ub.pop(i)
            lb.pop(i)
        x_relaxed, lower_bound = solver(H, y, bounds=(lb, ub))
        score_right = lower_bound - node['lower_bound']

        score = max(score_left, 1e-6) * max(score_right, 1e-6)
        if best_score is None or score > best_score:
            best_score = score
            best_i = i
    return best_i


def branch_and_bound_solver(H, y, strong=False):
    def branch(upper_bound, sorted_nodes, strong=False):
        node = sorted_nodes.pop(0)
        # print(node)
        # print(upper_bound)
        # print('-'*20)
        if np.allclose(node['x_relaxed'], node['x_rounded']):
            for i, x in zip(node['var_map'], node['x_rounded']):
                node['ec'][i] = x
            return upper_bound, np.array([v for k, v in sorted(node['ec'].items())])
        if strong:
            i_to_branch = strong_branching(node)
        else:
            i_to_branch = np.abs(node['x_relaxed'] - node['x_rounded']).argmax()

        left_node = deepcopy(node)
        ub_i = np.floor(node['x_relaxed'][i_to_branch])
        left_node['ub'][i_to_branch] = ub_i
        if np.isclose(left_node['lb'][i_to_branch], left_node['ub'][i_to_branch]):
            left_node['y'] = node['y'] - node['H'][:, i_to_branch] * ub_i
            left_node['H'] = np.concatenate([node['H'][:, :i_to_branch], node['H'][:, i_to_branch+1:]], axis=1)

            left_node['lb'].pop(i_to_branch)
            left_node['ub'].pop(i_to_branch)

            left_node['ec'][node['var_map'][i_to_branch]] = ub_i
            left_node['var_map'].pop(i_to_branch)
        upper_bound = bound(left_node, upper_bound, sorted_nodes)

        right_node = deepcopy(node)
        lb_i = np.ceil(node['x_relaxed'][i_to_branch])
        right_node['lb'][i_to_branch] = lb_i
        if np.isclose(right_node['ub'][i_to_branch], right_node['lb'][i_to_branch]):
            right_node['y'] = node['y'] - node['H'][:, i_to_branch] * lb_i
            right_node['H'] = np.concatenate([node['H'][:, :i_to_branch], node['H'][:, i_to_branch+1:]], axis=1)

            right_node['ub'].pop(i_to_branch)
            right_node['lb'].pop(i_to_branch)

            right_node['ec'][node['var_map'][i_to_branch]] = lb_i
            right_node['var_map'].pop(i_to_branch)
        upper_bound = bound(right_node, upper_bound, sorted_nodes)

        return upper_bound, None

    def bound(node, upper_bound, sorted_nodes):
        x_relaxed, lower_bound = solver(node['H'], node['y'],
                                        bounds=(node['lb'], node['ub']))

        x_relaxed = np.round(x_relaxed, decimals=8)
        x_rounded = x_relaxed.astype(int)
        unbound = ((node['H'] @ x_rounded - node['y']) ** 2).sum()

        # print(upper_bound, x_relaxed, x_rounded, lower_bound)
        # print('-'*20)

        for i, n in enumerate(sorted_nodes):
            if unbound < n['lower_bound']:
                sorted_nodes.pop(i)

        if np.isclose(lower_bound, unbound):
            upper_bound = max(upper_bound, unbound)

        node['x_relaxed'] = x_relaxed
        node['x_rounded'] = x_rounded
        node['lower_bound'] = lower_bound
        sorted_nodes.add(node)

        nonlocal total_nodes
        total_nodes += 1
        return upper_bound

    dim = H.shape[1]
    total_nodes = 0
    lb = [-np.inf] * dim
    ub = [np.inf] * dim
    ec = {}
    var_map = list(range(dim))

    relaxed_solution = lsq_linear(H, y)

    x_relaxed = relaxed_solution.x
    x_rounded = x_relaxed.astype(int)
    lower_bound = relaxed_solution.cost
    upper_bound = ((H @ x_rounded - y) ** 2).sum()

    sorted_nodes = SortedList(key=lambda x: x['lower_bound'])
    sorted_nodes.add({'x_relaxed': x_relaxed, 'x_rounded': x_rounded,
                      'lb': lb, 'ub': ub, 'ec': ec, 'var_map': var_map,
                      'H': H, 'y': y,
                      'lower_bound': lower_bound})
    while True:
        upper_bound, solution = branch(upper_bound, sorted_nodes, strong)
        if solution is not None:
            break
    return upper_bound, solution, total_nodes


def test_branch_and_bound(dim, set_size, n_tests=100, strong=False):
    cnt = 0
    cnt_nodes = 0
    for _ in tqdm(range(n_tests)):
        H = np.random.uniform(-set_size, set_size, size=(dim, dim))
        x = np.random.randint(-set_size, set_size+1, size=dim)
        y = H @ x + np.random.normal(0, 0.01, size=dim)

        upper_bound, solution, total_nodes = branch_and_bound_solver(H, y, strong)

        unbound_true = ((H @ x - y) ** 2).sum()
        unbound_pred = ((H @ solution - y) ** 2).sum()
        if not (unbound_pred <= unbound_true):
            continue
        cnt += 1
        cnt_nodes += total_nodes
    return cnt / n_tests, cnt_nodes / cnt
