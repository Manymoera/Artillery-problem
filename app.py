import webbrowser
from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import linear_sum_assignment

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/solve', methods=['POST'])
def solve():
    data = request.json
    C = np.array(data['matrix'], dtype=float)
    k = float(data.get('k', 2.0))
    n = C.shape[0]

    # 1. Формируем модифицированную матрицу C_tilde (Задача 2)
    # Элемент = мощь в текущий + мощь в следующий период [cite: 34, 43-47]
    C_tilde = np.zeros((n, n))
    for j in range(n):
        for i in range(n):
            if j < n - 1:
                C_tilde[i, j] = C[i, j] + C[i, j+1]
            else:
                C_tilde[i, j] = C[i, j]

    # 2. Решаем задачу о назначениях (максимизация S6) [cite: 51]
    row_ind, col_ind = linear_sum_assignment(C_tilde, maximize=True)
    
    S6_max = float(C_tilde[row_ind, col_ind].sum())
    
    # 3. Вычисляем S5 по формуле (58) 
    total_sum_C = float(np.sum(C))
    S5_min = total_sum_C - ((k - 1) / k) * S6_max

    # Подготовка расписания (какие ячейки подсветить)
    selections = [{"row": int(r), "col": int(c)} for r, c in zip(row_ind, col_ind)]

    return jsonify({
        "s5": round(S5_min, 2),
        "s6": round(S6_max, 2),
        "selections": selections
    })

if __name__ == '__main__':
    webbrowser.open("http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000)