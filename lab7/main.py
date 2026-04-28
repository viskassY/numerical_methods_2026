import numpy as np

# 1. Генерація матриці A
def generate_matrix(n):
    A = np.random.rand(n, n)

    # діагональне переважання
    for i in range(n):
        A[i][i] = sum(abs(A[i])) + 1

    return A

# 2. Запис у файл
def write_matrix(filename, A):
    np.savetxt(filename, A)

def write_vector(filename, b):
    np.savetxt(filename, b)

# 3. Зчитування
def read_matrix(filename):
    return np.loadtxt(filename)

def read_vector(filename):
    return np.loadtxt(filename)

# 4. Множення матриці на вектор
def mat_vec(A, x):
    n = len(x)
    b = np.zeros(n)

    for i in range(n):
        for j in range(n):
            b[i] += A[i][j] * x[j]

    return b

# 5. Норма вектора (максимальна)
def norm_vec(x):
    return np.max(np.abs(x))


# 6. Норма матриці (рядкова)
def norm_mat(A):
    return np.max(np.sum(np.abs(A), axis=1))


# 7. Метод простої ітерації
def simple_iteration(A, b, x0, eps):
    n = len(b) # кількість рівнянь
    x = x0.copy()

    tau = 1 / norm_mat(A)   # беремо допустиме τ

    for k in range(100000):
        x_new = x - tau * (np.dot(A, x) - b)
        # критерій зупинки
        if norm_vec(x_new - x) < eps:
            return x_new, k

        x = x_new

    return x, k


# 8. Метод Якобі
def jacobi(A, b, x0, eps):
    n = len(b)
    x = x0.copy()

    for k in range(100000):
        x_new = np.zeros(n)

        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        if norm_vec(x_new - x) < eps:
            return x_new, k

        x = x_new

    return x, k


# 9. Метод Зейделя
def seidel(A, b, x0, eps):
    n = len(b)
    x = x0.copy()

    for k in range(100000):
        x_new = x.copy()

        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i)) # нові значення
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))# старі значення

            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        if norm_vec(x_new - x) < eps:
            return x_new, k

        x = x_new

    return x, k


# ОСНОВНА ЧАСТИНА
n = 100
A = generate_matrix(n)

x_true = np.full(n, 2.5)

b = mat_vec(A, x_true)

write_matrix("A.txt", A)
write_vector("B.txt", b)

A = read_matrix("A.txt")
b = read_vector("B.txt")

# початкове наближення
x0 = np.ones(n)

eps = 1e-14

# розв'язки
x_si, k_si = simple_iteration(A, b, x0, eps)
x_j, k_j = jacobi(A, b, x0, eps)
x_s, k_s = seidel(A, b, x0, eps)

print("Норма матриці А: ", norm_mat(A))

# вивід (скільки кроків потрібно, щоб знайти розв’язок з точністю ε)
print("Кількість ітерацій\n")
print("Метод простої ітерації:", k_si)
print("Розв'язок (перші 5):", x_si[:5])
print("Похибка:", norm_vec(x_si - x_true))

print("\nМетод Якобі:", k_j)
print("Розв'язок (перші 5):", x_j[:5])
print("Похибка:", norm_vec(x_j - x_true))

print("\nМетод Зейделя:", k_s)
print("Розв'язок (перші 5):", x_s[:5])
print("Похибка:", norm_vec(x_s - x_true))