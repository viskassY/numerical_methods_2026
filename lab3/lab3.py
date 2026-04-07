import numpy as np
import matplotlib.pyplot as plt
import csv

# -------------------------------
# 1. Зчитування даних з CSV
# -------------------------------
def read_csv(data):
    x = []
    y = []
    with open(data, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['Month']))
            y.append(float(row['Temp']))
    return np.array(x), np.array(y)


# -------------------------------
# 2. Формування матриці A
# -------------------------------
def form_matrix(x, m):
    n = m + 1
    A = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A[i][j] = np.sum(x ** (i + j))
    return A


# -------------------------------
# 3. Формування вектора b
# -------------------------------
def form_vector(x, y, m):
    n = m + 1
    b = np.zeros(n)
    for i in range(n):
        b[i] = np.sum(y * (x ** i))
    return b


# -------------------------------
# 4. Метод Гауса
# -------------------------------
def gauss_solve(A, b):
    n = len(b)
    # Прямий хід
    for k in range(n):
        # Пошук головного елемента
        max_row = np.argmax(np.abs(A[k:, k])) + k

        # Перестановка рядків
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        # Виключення
        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Зворотній хід
    x_sol = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x_sol[i] = (b[i] - np.sum(A[i, i+1:] * x_sol[i+1:])) / A[i, i]

    return x_sol


# -------------------------------
# 5. Обчислення полінома
# -------------------------------
def polynomial(x, coef):
    y = np.zeros_like(x, dtype=float)
    for i in range(len(coef)):
        y += coef[i] * (x ** i)
    return y


# -------------------------------
# 6. Дисперсія
# -------------------------------
def variance(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)


# -------------------------------
# 7. мейн
# -------------------------------
def main():
    x, y = read_csv("data.csv")
    max_degree = 8
    variances = []

    # Пошук оптимального степеня
    for m in range(1, max_degree + 1):
        A = form_matrix(x, m)
        b = form_vector(x, y, m)
        coef = gauss_solve(A.copy(), b.copy())

        y_approx = polynomial(x, coef)
        var = variance(y, y_approx)
        variances.append(var)

        print(f"Степінь {m}: дисперсія = {var:.4f}")

    optimal_m = np.argmin(variances) + 1
    print("\nОптимальний степінь:", optimal_m)

    # Побудова оптимального полінома
    A = form_matrix(x, optimal_m)
    b = form_vector(x, y, optimal_m)
    coef = gauss_solve(A.copy(), b.copy())

    # 2. ТАБУЛЯЦІЯ для плавних графіків (пункт 4 методички)
    x_fine = np.linspace(x.min(), x.max(), 200)
    y_fine = polynomial(x_fine, coef)

    # Прогноз
    x_future = np.array([25, 26, 27])
    y_future = polynomial(x_future, coef)

    print("\nПрогноз температур:")
    for xi, yi in zip(x_future, y_future):
        print(f"Місяць {xi}: {yi:.2f}")

    # Похибка у вузлах (як в методичці)
    y_at_nodes = polynomial(x, coef)
    error_values = y - y_at_nodes

    # -------------------------------
    # Графіки
    # -------------------------------

    # 1. Дані + апроксимація
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.scatter(x, y, label="Дані")
    plt.plot(x_fine, y_fine, label="Апроксимація")
    plt.xlabel("Місяць")
    plt.ylabel("Температура, °C")
    plt.legend()
    plt.title("Апроксимація поліномом")
    plt.grid()

    # 2. Дисперсія
    plt.subplot(2, 2, 2)
    plt.plot(range(1, max_degree + 1), variances, marker='o')
    plt.title("Дисперсія від степеня полінома")
    plt.xlabel("Степінь")
    plt.ylabel("Дисперсія")
    plt.grid()

    # 3. Похибка
    plt.subplot(2, 2, 3)
    plt.plot(x, error_values, 'o-', color='green', label="ε(x)")
    plt.axhline(0)  # вісь 0
    plt.xlabel("Місяць")
    plt.ylabel("Похибка")
    plt.title("Похибка апроксимації")
    plt.legend()
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.scatter(x, y, label="Дані")
    plt.plot(x_fine, y_fine, label="Модель")
    plt.scatter(x_future, y_future, marker='x', label="Прогноз")
    plt.title("Прогноз")
    plt.xlabel("Місяць")
    plt.ylabel("Температура")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()