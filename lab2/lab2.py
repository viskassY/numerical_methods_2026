import numpy as np
import matplotlib.pyplot as plt
import csv

# Зчитування
def read_data(filename):
    x = []
    y = []
    with open(filename, 'r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['tasks']))
            y.append(float(row['cost']))
    return np.array(x), np.array(y)


# Таблиця розділених різниць
def divided_differences(x, y):
    n = len(x)
    table = np.zeros((n, n))

    table[:,0] = y
    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j] - x[i])
    return table


# Поліном Ньютона
def newton_polynomial(x, table, x_val):

    n = len(x)
    result = table[0][0]
    product = 1

    for i in range(1, n):
        product *= (x_val - x[i-1])
        result += table[0][i] * product
    return result

# Таблиця скінченних різниць
def finite_differences(y):

    n = len(y)
    table = np.zeros((n, n))
    table[:,0] = y

    for j in range(1, n):
        for i in range(n-j):
            table[i][j] = table[i+1][j-1] - table[i][j-1]
    return table

# Факторіальний поліном
def factorial_polynomial(x, y, x_val):

    n = len(x)
    #  рівномірна сітка
    x_uniform = np.linspace(min(x), max(x), n)
    # інтерполяція значень
    y_uniform = np.interp(x_uniform, x, y)
    h = x_uniform[1] - x_uniform[0]
    diff = finite_differences(y_uniform)
    t = (x_val - x_uniform[0]) / h
    result = y_uniform[0]

    term = 1
    factorial = 1

    #рахується поліном
    for i in range(1, n):
        term *= (t - (i-1))
        factorial *= i
        result += (term * diff[0][i]) / factorial
    return result


# MAIN

x, y = read_data("data.csv")
print("\nДані:")
for i in range(len(x)):
    print(x[i], y[i])


# таблиця розділених різниць
table = divided_differences(x, y)

print("\nТаблиця розділених різниць:\n")
for row in table:
    print(["{:.6f}".format(v) for v in row])

# Прогноз
x_pred = 15000

newton_pred = newton_polynomial(x, table, x_pred)
factorial_pred = factorial_polynomial(x, y, x_pred)
print("\nПрогноз для 15000 tasks:\n")
print("Newton =", newton_pred)
print("Factorial =", factorial_pred)
error = abs(newton_pred - factorial_pred)
print("\nРізниця методів =", error)

# Табуляція
xs = np.linspace(min(x), max(x), 300)
newton_vals = [newton_polynomial(x, table, xi) for xi in xs]
factorial_vals = [factorial_polynomial(x, y, xi) for xi in xs]

# Дослідження вузлів
def node_experiment(nodes):

    x_new = np.linspace(min(x), max(x), nodes)
    y_new = np.interp(x_new, x, y)

    table_new = divided_differences(x_new, y_new)
    vals = [newton_polynomial(x_new, table_new, xi) for xi in xs]

    return np.array(vals)

vals5 = node_experiment(5)
vals10 = node_experiment(10)
vals20 = node_experiment(20)

# ГРАФІКИ
plt.figure(figsize=(12,10))

# 1
plt.subplot(2,2,1)

plt.scatter(x, y, color='blue', label="Дані")
plt.plot(xs, newton_vals, color='red', label="Newton")

plt.title("Інтерполяція Ньютона")
plt.xlabel("Tasks")
plt.ylabel("Cost")
plt.legend()
plt.grid()


# 2
plt.subplot(2,2,2)

plt.scatter(x, y)
plt.plot(xs, factorial_vals, color='green')

plt.title("Факторіальний поліном")
plt.xlabel("Tasks")
plt.ylabel("Cost")
plt.grid()


# 3
plt.subplot(2,2,3)

plt.plot(xs, vals5, label="5 вузлів")
plt.plot(xs, vals10, label="10 вузлів")
plt.plot(xs, vals20, label="20 вузлів")

plt.title("Вплив кількості вузлів")
plt.legend()
plt.grid()


# 4
plt.subplot(2,2,4)

ref = np.interp(xs, x, y)

plt.plot(xs, abs(vals5-ref), label="5")
plt.plot(xs, abs(vals10-ref), label="10")
plt.plot(xs, abs(vals20-ref), label="20")

plt.title("Похибки")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()