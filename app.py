from flask import Flask, render_template, request, jsonify
import numpy as np
from scipy.optimize import linear_sum_assignment
import webbrowser

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    data = request.json
    # Матрица разведданных C (n x n)
    C = np.array(data['matrix'], dtype=float)
    n = C.shape[0]
    
    # Параметры из UI
    k = float(data.get('k', 2.0))
    alpha_base = float(data.get('intensity', 0.15))
    daily_r = float(data.get('total_r', 40))
    rep_strat = data.get('replenish_strat', 'patching')
    strike_strat = data.get('strike_strat', 'task2_math')

    # --- 1. ПРЕДВАРИТЕЛЬНЫЙ РАСЧЕТ ДЛЯ МАТЕМАТИЧЕСКОЙ СТРАТЕГИИ ---
    # Определение задачи (Task 1 или Task 2)
    current_task = data.get('task_type', 'task2')

    schedule = {}
    if strike_strat == 'task2_math':
        if current_task == 'task1':
            # ЗАДАЧА 1: Оптимизация по исходной матрице C
            cost_matrix = C
        else:
            # ЗАДАЧА 2: Оптимизация по сумме двух дней
            cost_matrix = np.zeros((n, n))
            for j in range(n):
                for i in range(n):
                    cost_matrix[i, j] = (C[i, j] + C[i, j+1]) if j < n-1 else C[i, j]
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)
        schedule = {int(j): int(row_ind[np.where(col_ind == j)[0][0]]) for j in range(n)}

    # --- 2. ДИНАМИЧЕСКАЯ СИМУЛЯЦИЯ ---
    history = []
    visual_selections = [] # Список для подсветки ячеек в интерфейсе
    current_powers = np.copy(C[:, 0]) # Текущие силы (начинаем с 1-го дня)
    initial_ref = np.copy(C[:, 0])   # Ориентир для пополнения (patching)
    total_s5 = 0
    active_k = {} # Словарь {индекс_подразделения: оставшиеся_дни_эффекта}
    hit_units = [] # Список уже пораженных подразделений

    for j in range(n):
        # ВЫБОР ЦЕЛИ (Target Selection)
        if strike_strat == 'task2_math':
            target = schedule[j]
        else:
            # Ищем доступные подразделения (по которым еще не стреляли за всю операцию)
            available = [i for i in range(n) if i not in hit_units]
            if not available:
                target = -1
            elif strike_strat == 'logistics_collapse':
                # Цель: юниты с минимальной мощью (где эффект пополнения будет нивелирован k)
                target = available[np.argmin(current_powers[available])]
            elif strike_strat == 'focused_wear':
                # Цель: самые сильные юниты в моменты пиковой интенсивности
                intensity_now = alpha_base * (1 + np.sin(np.pi * j / n))
                if intensity_now > alpha_base:
                    target = available[np.argmax(current_powers[available])]
                else:
                    target = available[np.argmin(current_powers[available])]
            else: # max_power (Жадный алгоритм)
                target = available[np.argmax(current_powers[available])]
            
            if target != -1:
                hit_units.append(target)

        # Регистрация выбора для визуализации (подсветка зеленым)
        if target != -1:
            visual_selections.append({"row": int(target), "col": j})
            # Длительность эффекта зависит от задачи
            active_k[target] = 1 if current_task == 'task1' else 2

        # РАСЧЕТ ТЕКУЩЕЙ ЭФФЕКТИВНОЙ МОЩНОСТИ (с учетом k)
        eff_powers = np.copy(current_powers)
        for idx in list(active_k.keys()):
            eff_powers[idx] /= k

        # ШАГ УБЫВАНИЯ (Дневной бой)
        # Интенсивность боя меняется по времени (модель Ланчестера) 
        intensity = alpha_base * (1 + np.sin(np.pi * j / n))
        losses = eff_powers * (1 - np.exp(-intensity))
        current_powers -= losses
        
        # ШАГ ПОПОЛНЕНИЯ (Вечерние резервы)
        if rep_strat == 'patching':
            # Распределение пропорционально потерям относительно 1-го дня
            gap = np.maximum(0, initial_ref - current_powers)
            if gap.sum() > 0:
                current_powers += (gap / gap.sum()) * daily_r
        else: # strengthening
            # Распределение пропорционально текущей выжившей мощи
            if current_powers.sum() > 0:
                current_powers += (current_powers / current_powers.sum()) * daily_r

        # ОБНОВЛЕНИЕ ТАЙМЕРОВ ЭФФЕКТА k
        for idx in list(active_k.keys()):
            active_k[idx] -= 1
            if active_k[idx] <= 0:
                del active_k[idx]

        # Запись истории дня
        day_sum = eff_powers.sum()
        total_s5 += day_sum
        history.append({
            "day": j + 1,
            "target": int(target) + 1 if target != -1 else "Нет",
            "powers": [round(p, 1) for p in eff_powers],
            "sum": round(day_sum, 2)
        })

    return jsonify({
        "history": history,
        "total_s5": round(total_s5, 2),
        "selections": visual_selections # Теперь передается для всех стратегий
    })

if __name__ == '__main__':
    # Автоматическое открытие браузера при запуске
    webbrowser.open("http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)