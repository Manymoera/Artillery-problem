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
    C = np.array(data['matrix'], dtype=float)
    n = C.shape[0]
    k = float(data.get('k', 2.0))
    task_type = data.get('task_type', 'task2')
    
    # Параметры для динамики (только для Задачи 2)
    alpha_base = float(data.get('intensity', 0.15))
    daily_r = float(data.get('total_r', 40))
    rep_strat = data.get('replenish_strat', 'patching')
    strike_strat = data.get('strike_strat', 'task2_math')

    history = []
    visual_selections = []
    
    # ЛОГИКА ЗАДАЧИ 1 (Строго по документу: Статика)
    if task_type == 'task1':
        # 1. Поиск оптимальных назначений по исходной матрице C
        row_ind, col_ind = linear_sum_assignment(C, maximize=True)
        assignments = {int(j): int(row_ind[np.where(col_ind == j)[0][0]]) for j in range(n)}
        
        total_s = 0
        for j in range(n):
            target = assignments[j]
            visual_selections.append({"row": target, "col": j})
            
            # Формируем состояние сил на день j (без влияния прошлых дней)
            day_powers = []
            day_sum = 0
            for i in range(n):
                val = C[i, j]
                if i == target:
                    val /= k # Эффект k только в день удара и только для цели
                day_powers.append(round(val, 2))
                day_sum += val
            
            total_s += day_sum
            history.append({
                "day": j + 1,
                "target": target + 1,
                "powers": day_powers,
                "sum": round(day_sum, 2)
            })
        
        s6_final = history[-1]['sum'] 

        return jsonify({
            "history": history,
            "total_s5": round(total_s, 2),
            "total_s6": round(s6_final, 2),
            "selections": visual_selections
        })

    # ЛОГИКА ЗАДАЧИ 2
    else:
        # 1. Предварительный расчет для математической стратегии
        schedule = {}
        if strike_strat == 'task2_math':
            C_tilde = np.zeros((n, n))
            for j in range(n):
                for i in range(n):
                    C_tilde[i, j] = (C[i, j] + C[i, j+1]) if j < n-1 else C[i, j]
            r_ind, c_ind = linear_sum_assignment(C_tilde, maximize=True)
            schedule = {int(j): int(r_ind[np.where(c_ind == j)[0][0]]) for j in range(n)}

        current_powers = np.copy(C[:, 0])
        initial_ref = np.copy(C[:, 0])
        total_s5 = 0
        active_k = {} 
        hit_units = []

        for j in range(n):
            if strike_strat == 'task2_math':
                target = schedule[j]
            else:
                available = [i for i in range(n) if i not in hit_units]
                if not available: target = -1
                elif strike_strat == 'logistics_collapse':
                    target = available[np.argmin(current_powers[available])]
                elif strike_strat == 'focused_wear':
                    intensity_now = alpha_base * (1 + np.sin(np.pi * j / n))
                    target = available[np.argmax(current_powers[available])] if intensity_now > alpha_base else available[np.argmin(current_powers[available])]
                else: # max_power
                    target = available[np.argmax(current_powers[available])]
                
                if target != -1: hit_units.append(target)

            if target != -1:
                visual_selections.append({"row": int(target), "col": j})
                active_k[target] = 2

            # ПРИМЕНЕНИЕ ЭФФЕКТА И БОЯ
            eff_powers = np.copy(current_powers)
            for idx, ttl in active_k.items():
                eff_powers[idx] /= k

            intensity = alpha_base * (1 + np.sin(np.pi * j / n))
            current_powers -= eff_powers * (1 - np.exp(-intensity))
            
            # ПОПОЛНЕНИЕ РЕЗЕРВОВ
            if rep_strat == 'patching':
                gap = np.maximum(0, initial_ref - current_powers)
                if gap.sum() > 0: current_powers += (gap / gap.sum()) * daily_r
            else:
                if current_powers.sum() > 0: current_powers += (current_powers / current_powers.sum()) * daily_r

            # Обновление таймеров
            active_k = {idx: ttl-1 for idx, ttl in active_k.items() if ttl > 1}

            day_sum = eff_powers.sum()
            total_s5 += day_sum
            history.append({
                "day": j + 1,
                "target": int(target) + 1 if target != -1 else "Нет",
                "powers": [round(p, 1) for p in eff_powers],
                "sum": round(day_sum, 2)
            })

        s6_final = current_powers.sum()

        return jsonify({
            "history": history,
            "total_s5": round(total_s5, 2),
            "total_s6": round(s6_final, 2), # Добавили S6
            "selections": visual_selections
        })

if __name__ == '__main__':
    # Автоматическое открытие браузера при запуске
    webbrowser.open("http://127.0.0.1:8000")
    app.run(host="127.0.0.1", port=8000, debug=True)