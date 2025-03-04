import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('TkAgg')  # 设置后端为 TkAgg
import matplotlib.pyplot as plt


"""本程序使用了一些简单的步骤实现了完成课本程序的复现，本程序使用了类似于全局变量的操作，使得用户可以更改函数、步长、初值进行其他ODE方程的求解。"""

# 解析解定义
def exact_solution(x):
    return (1 / 2) * (x * x + 2) * (np.exp(-x))

#欧拉法
def euler_method(f, x0, y0, h, x_end):
    """使用欧拉法求解ODE：y'=x*np.exp(-x)-y"""
    x_values = np.arange(x0, x_end + h, h)  # 生成x值
    y_values = np.zeros(len(x_values))  # 预分配y值
    y_values[0] = y0  # 初值
    errors = np.zeros(len(x_values))#误差
    # 循环输出
    for i in range(1, len(x_values)):
        y_values[i] = y_values[i - 1] + h * f(x_values[i - 1], y_values[i - 1])
        y_exact = exact_solution(x_values[i])
        errors[i] = np.abs(y_values[i] - y_exact)


    return x_values, y_values,errors

# 定义ODE：dy/dx=x*np.e(-x)-y
def f(x, y):
    return x * np.exp(-x) - y

#设置初值
x0,y0=0,1
x_end=1
step_size=[0.1,0.05,0.01]

#画图
plt.figure(figsize=(8,6))

#绘制解析解
x_exact=np.linspace(x0,x_end,100)
y_exact=exact_solution(x_exact)
plt.plot(x_exact,y_exact,label="Exact Solution",linestyle="dashed",color='black')

# 使用欧拉法求数值解
colors=['b','g','r']

#创建excel
with pd.ExcelWriter("euler_results.xlsx", engine="openpyxl") as writer:
    for h,color in zip(step_size,colors):
        #计算数值解
        x_vals,y_vals,errors=euler_method(f,x0,y0,h,x_end)

        #画图
        plt.plot(x_vals,y_vals,marker='o',linestyle='-',color=color,label=f"Euler h={h}")

        #存入DataFrame
        df=pd.DataFrame({
            "x":x_vals,
            "y_exact":exact_solution(x_vals),
            "y_numerical":y_vals,
            "error":errors
        })

        #写入Excel，每种步长的数据存入不同的sheet
        df.to_excel(writer,sheet_name=f"h={h}",index=False)

        #打印结果
        print(f"\n步长 h={h} 的计算结果：")
        print(df)

#设置图例
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Euler Method Approximation with Different Step Sizes")
plt.grid()
plt.show()

print("\n数据已成功保存到 euler_results.xlsx，每个步长数据存入不同 Sheet")