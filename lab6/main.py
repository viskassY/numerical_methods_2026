import random

# 1. Генерація матриці A і вектора B
def generate_data(n=100):
    A = [[random.uniform(1, 10) for _ in range(n)] for _ in range(n)]
    x_true = [2.5 for _ in range(n)]

    # b_i = sum(a_ij * x_j)
    B = []
    for i in range(n):
        s = 0
        for j in range(n):
            s += A[i][j] * x_true[j]
        B.append(s)

    return A, B


def write_matrix(A, filename):
    with open(filename, "w") as f:
        for row in A:
            f.write(" ".join(map(str, row)) + "\n")


def write_vector(B, filename):
    with open(filename, "w") as f:
        f.write(" ".join(map(str, B)))


# 2. Зчитування даних
def read_matrix(filename):
    A = []
    with open(filename, "r") as f:
        for line in f:
            A.append(list(map(float, line.split())))
    return A


def read_vector(filename):
    with open(filename, "r") as f:
        return list(map(float, f.readline().split()))


# 2. LU-розклад
def lu_decomposition(A):
    n = len(A)
    L = [[0]*n for _ in range(n)]
    U = [[0]*n for _ in range(n)]

    for i in range(n):
        U[i][i] = 1

    for k in range(n):
        # L
        for i in range(k, n):
            s = 0
            for j in range(k):
                s += L[i][j] * U[j][k]
            L[i][k] = A[i][k] - s

        # U
        for i in range(k+1, n):
            s = 0
            for j in range(k):
                s += L[k][j] * U[j][i]
            U[k][i] = (A[k][i] - s) / L[k][k]

    return L, U


def write_lu(L, U, filename):
    with open(filename, "w") as f:
        f.write("L:\n")
        for row in L:
            f.write(" ".join(map(str, row)) + "\n")

        f.write("\nU:\n")
        for row in U:
            f.write(" ".join(map(str, row)) + "\n")


# 2. Допоміжні функції
def mat_vec_mult(A, x):
    n = len(A)
    result = [0]*n
    for i in range(n):
        for j in range(n):
            result[i] += A[i][j] * x[j]
    return result


def vector_norm(v):
    return max(abs(x) for x in v)

# 3. Розв’язок через LU
def solve_lu(L, U, B):
    n = len(B)

    # LZ = B прямий хід (зверху вниз)
    Z = [0]*n
    for i in range(n):
        s = 0
        for j in range(i):
            s += L[i][j] * Z[j]
        Z[i] = (B[i] - s) / L[i][i]

    # UX = Z (знизу вверх)
    X = [0]*n
    for i in range(n-1, -1, -1):
        s = 0
        for j in range(i+1, n):
            s += U[i][j] * X[j]
        X[i] = Z[i] - s

    return X


# 4. Оцінка точності
def calc_error(A, X, B):
    eps = 0
    AX = mat_vec_mult(A, X)

    for i in range(len(A)):
        eps = max(eps, abs(AX[i] - B[i]))

    return eps


# 5. Ітераційне уточнення
def iterative_refinement(A, L, U, B, X, eps_target=1e-14, max_iter=50):
    iterations = 0

    while True:
        AX = mat_vec_mult(A, X)
        R = [B[i] - AX[i] for i in range(len(B))] #вектор помилки R
        dX = solve_lu(L, U, R)

        X_new = [X[i] + dX[i] for i in range(len(X))] #знайдену поправку додаємо до X

        # перевірка
        if vector_norm(dX) <= eps_target:
            break

        if iterations >= max_iter:
            print("Досягнуто максимум ітерацій")
            break

        X = X_new
        iterations += 1

    return X_new, iterations



# MAIN
def main():
    # 1
    A, B = generate_data(100)
    write_matrix(A, "A.txt")
    write_vector(B, "B.txt")

    # 2
    A = read_matrix("A.txt")
    B = read_vector("B.txt")

    L, U = lu_decomposition(A)
    write_lu(L, U, "LU.txt")

    # 3
    X = solve_lu(L, U, B)
    print("Розв'язок(X):")
    print(X)

    # 4
    eps = calc_error(A, X, B)
    print("Похибка:", eps)

    # 5
    X_refined, iters = iterative_refinement(A, L, U, B, X)
    print("Кількість ітерацій:", iters)
    print("Уточнений розв'язок:")
    print(X_refined)


if __name__ == "__main__":
    main()