from typing import List
import numpy as np
import matplotlib.pyplot as plt

# Дані
matrix = np.array([
    [10, 1, 5, 10, 3, 8, 9, 2, 5, 4, 7, 7],
    [6, 1, 8, 5, 2, 5, 4, 2, 2, 1, 5, 7],
    [5, 10, 5, 8, 4, 6, 7, 5, 7, 7, 7, 3],
    [2, 4, 9, 8, 4, 10, 2, 8, 6, 10, 9, 7],
    [8, 8, 3, 4, 9, 5, 8, 9, 9, 2, 1, 3],
    [3, 8, 7, 5, 2, 6, 1, 2, 3, 5, 1, 7],
    [6, 4, 5, 7, 2, 6, 9, 2, 5, 4, 1, 10],
    [10, 9, 10, 1, 5, 2, 6, 2, 10, 2, 6, 10],
    [3, 8, 7, 9, 2, 5, 6, 5, 3, 9, 8, 7],
    [1, 4, 4, 4, 9, 4, 7, 1, 7, 5, 8, 8],
    [3, 2, 3, 2, 1, 5, 3, 1, 6, 3, 3, 6],
    [5, 9, 9, 3, 9, 10, 5, 8, 9, 9, 10, 10],
    [2, 9, 9, 8, 1, 4, 2, 8, 6, 3, 8, 4],
    [7, 1, 2, 8, 9, 10, 3, 7, 2, 7, 3, 6],
    [2, 2, 5, 4, 10, 5, 4, 7, 7, 3, 1, 7]
])

weights = np.array([8, 9, 10, 2, 9, 4, 8, 8, 4, 8, 1, 8])
weights_sum = np.sum(weights)

# Порогові значення індексів узгодження та неузгодження
c_threshold = 0.744
d_threshold = 0.280

alternatives_num, criteria_num = matrix.shape
weights_max_dif = [max(matrix[:, col]) - min(matrix[:, col]) for col in range(criteria_num)]

# Допоміжні функції
def calc_concordance(el1: int, el2: int):
    numerator = sum(weights[col] for col in range(criteria_num) if matrix[el1, col] >= matrix[el2, col])
    return round(numerator / weights_sum, 3)

def calc_discordance(el1: int, el2: int):
    max_numerator = 0
    max_denominator = 0
    for col in range(criteria_num):
        diff = matrix[el2, col] - matrix[el1, col]
        if diff > 0:
            max_numerator = max(max_numerator, diff * weights[col])
            max_denominator = max(max_denominator, weights_max_dif[col] * weights[col])
    return round(max_numerator / max_denominator, 3) if max_denominator > 0 else 0

def dfs(used: List, checked: List, row: int, matrix: List[List[int]]):
    checked.append(row)
    for col in range(alternatives_num):
        if matrix[row][col] == 0:
            continue
        if matrix[row][col] == 1 and col not in checked:
            if dfs(used, checked, col, matrix):
                return True
        elif col in checked:
            return True
    checked.remove(row)
    used.append(row)
    return False

def check_if_cyclic(matrix: List[List[int]]):
    used = []
    for node in range(alternatives_num):
        if node not in used:
            if dfs(used, [], node, matrix):
                return True
    return False

def get_upper_contour_set(matrix: List[List[int]], node: int):
    return {i for i in range(alternatives_num) if matrix[i][node] == 1}

def get_Neumann_Morgenstern_S(matrix: List[List[int]]):
    S = []
    upper_sets = [get_upper_contour_set(matrix, i) for i in range(alternatives_num)]
    S0 = [i for i, up_set in enumerate(upper_sets) if not up_set]
    S.append(S0)
    while set(S[-1]) != set(range(alternatives_num)):
        Si = [i for i in range(alternatives_num) if upper_sets[i].issubset(S[-1])]
        S.append(Si)
    return S

def get_Neumann_Morgenstern_Q(matrix: List[List[int]], S: List[List[int]]):
    Q = [S[0]]
    upper_sets = [get_upper_contour_set(matrix, i) for i in range(alternatives_num)]
    for i in range(1, len(S)):
        Q.append(Q[-1].copy())
        for j in set(S[i]) - set(S[i-1]):
            if not set(upper_sets[j]).intersection(Q[i-1]):
                Q[i].append(j)
    return Q

def check_internal_stability(matrix: List[List[int]], arr: List[int]):
    return all(matrix[row][col] == 0 for row in arr for col in arr)

def check_external_stability(matrix: List[List[int]], arr: List[int]):
    for col in range(alternatives_num):
        if col not in arr:
            if not any(matrix[row][col] == 1 for row in arr):
                return False
    return True

def get_PIN(matrix: List[List[int]]):
    res_mat = []
    for row in range(alternatives_num):
        res_mat.append([])
        for col in range(alternatives_num):
            if matrix[row][col] == matrix[col][row] == 1:
                res_mat[row].append('I')
            elif matrix[row][col] == matrix[col][row] == 0:
                res_mat[row].append('N')
            elif matrix[row][col] == 1:
                res_mat[row].append('P')
            else:
                res_mat[row].append('-')
    return res_mat

def get_S(pin: List[List[str]], pin_str: str):
    S1 = [[col for col, val in enumerate(row) if val in pin_str] for row in pin]
    max_S = max(S1, key=len, default=[])
    return [i for i, row in enumerate(S1) if row == max_S]

# Основний алгоритм
concordance = [[calc_concordance(row, col) if row != col else 0 for col in range(alternatives_num)] for row in range(alternatives_num)]
discordance = [[calc_discordance(row, col) if row != col else 1 for col in range(alternatives_num)] for row in range(alternatives_num)]
relation_c = [[1 if concordance[row][col] >= c_threshold  else 0 for col in range(alternatives_num)] for row in range(alternatives_num)]
relation_d = [[1 if discordance[row][col] <= d_threshold else 0 for col in range(alternatives_num)] for row in range(alternatives_num)]
relation = [[1 if concordance[row][col] >= c_threshold and discordance[row][col] <= d_threshold else 0 for col in range(alternatives_num)] for row in range(alternatives_num)]

print('Матриця iндексiв узгодження С:')
for row in concordance:
    print(["{:.3f}".format(val) for val in row])

print('\nМатриця iндексiв неузгодження D:')
for row in discordance:
    print(["{:.3f}".format(val) for val in row])

print('\nПорiг:')
print(c_threshold, d_threshold, '\n')
print('\nC(Ai, Ak)≥ c˘')
for row in relation_c:
    print(row)
print('\nD(Ai, Ak)≤ d˘')
for row in relation_d:
    print(row)
print('\nВiдношення:')
for row in relation:
    print(row)

if not check_if_cyclic(relation):
    print('\nАциклічне: True')
    S = get_Neumann_Morgenstern_S(relation)
    Q = get_Neumann_Morgenstern_Q(relation, S)
    int_stab = check_internal_stability(relation, Q[-1])
    ext_stab = check_external_stability(relation, Q[-1])
    print('Внутрішня стабільність:', int_stab)
    print('Зовнішня стабільність:', ext_stab)
    neumann_morgenstern_set = [x + 1 for x in Q[-1]]
    print('Множина Неймана-Моргенштерна:', neumann_morgenstern_set)
else:
    print('\nАциклічне: False')
    pin = get_PIN(relation)
    for row in pin:
        print(row)
    print('\nResults:')
    print('k1 max el:', get_S(pin, 'NPI'))
    print('k2 max el:', get_S(pin, 'NP'))
    print('k3 max el:', get_S(pin, 'PI'))
    print('k4 max el:', get_S(pin, 'P'))
