import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt

"""本程序使用了一些简单的步骤实现了完成课本程序的复现，本程序使用了类似于全局变量的操作，使得用户可以更改函数、步长、初值进行其他ODE方程的求解。"""

# 解析解定义
def exact_solution(x):
    return (1 / 2) * (x * x + 2) * (np.exp(-x))


def euler_method(f, x0, y0, h, x_end):
    """使用欧拉法求解ODE：y'=x*np.exp(-x)-y"""
    x_values = np.arange(x0, x_end + h, h)  # 生成x值
    y_values = np.zeros(len(x_values))  # 预分配y值
    y_values[0] = y0  # 初值
    # 循环输出
    for i in range(1, len(x_values)):
        y_values[i] = y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1])
        y_exact = exact_solution(x_values[i])
        absolute_error = np.abs(y_values[i] - y_exact)
        print(
            f"x = {x_values[i]:.2f}, y_euler = {y_values[i]:.6f}, y_exact = {y_exact:.6f}, absolute_error = {absolute_error:.6f}")

    return x_values, y_values


# 定义ODE：dy/dx=x*np.e(-x)-y
def f(x, y):
    return x * np.exp(-x) - y


# 使用欧拉法求数值解
x_vals, y_vals = euler_method(f, x0=0, y0=1, h=0.1, x_end=1)

y_exact = exact_solution(x_vals)

plt.plot(x_vals, y_vals, label="Euler Approximation", marker='o')
plt.plot(x_vals, y_exact, label="Exact Solution", linestyle="dashed")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Euler Method vs. Exact Solution")
plt.show()