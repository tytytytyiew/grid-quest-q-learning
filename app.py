from flask import Flask, render_template, jsonify, request
import pickle
import numpy as np
from collections import defaultdict

app = Flask(__name__)


try:
    with open('q_model.pkl', 'rb') as f:
        model_data = pickle.load(f)

    GRID_SIZE = model_data.get('grid_size', 8)
    TREASURE_POS = tuple(model_data['treasure_pos'])
    WALLS = [tuple(w) for w in model_data['walls']]
    ACTIONS = model_data.get('actions', ['↑', '↓', '←', '→'])

    # ✔️ ВАЖНЫЙ ФИКС Q-TABLE
    q_table = defaultdict(lambda: np.zeros(len(ACTIONS)))

    for k, v in model_data['q_table'].items():
        q_table[tuple(k)] = np.array(v)

    print("✅ Модель загружена успешно")

except Exception as e:
    print("❌ Ошибка загрузки модели:", e)

    GRID_SIZE = 8
    TREASURE_POS = (7, 7)
    WALLS = []
    ACTIONS = ['↑', '↓', '←', '→']
    q_table = defaultdict(lambda: np.zeros(4))


ACTION_NAMES = {
    '↑': 'вверх',
    '↓': 'вниз',
    '←': 'влево',
    '→': 'вправо'
}


def get_best_action(row, col):
    state = (row, col)

    if state == TREASURE_POS or state in WALLS:
        return None

    return ACTIONS[np.argmax(q_table[state])]


def get_next_position(row, col, action):
    moves = {
        '↑': (-1, 0),
        '↓': (1, 0),
        '←': (0, -1),
        '→': (0, 1)
    }

    dr, dc = moves[action]
    nr, nc = row + dr, col + dc


    if nr < 0 or nr >= GRID_SIZE or nc < 0 or nc >= GRID_SIZE:
        return row, col

    if (nr, nc) in WALLS:
        return row, col

    return nr, nc


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/agent_move', methods=['POST'])
def agent_move():
    try:
        data = request.json
        current_pos = tuple(data['position'])

        if current_pos == TREASURE_POS:
            return jsonify({
                'success': True,
                'game_over': True,
                'message': '🎉 Уже на сокровище!'
            })

        action = get_best_action(current_pos[0], current_pos[1])

        if action is None:
            return jsonify({
                'success': False,
                'message': 'Нет хода'
            })

        new_pos = get_next_position(current_pos[0], current_pos[1], action)

        return jsonify({
            'success': True,
            'from': current_pos,
            'to': new_pos,
            'action': action,
            'found_treasure': new_pos == TREASURE_POS
        })

    except Exception as e:
        print("❌ ERROR agent_move:", e)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/auto_run', methods=['POST'])
def auto_run():
    try:
        data = request.json
        start = tuple(data['position'])

        path = [start]
        current = start

        for _ in range(100):
            if current == TREASURE_POS:
                break

            action = get_best_action(current[0], current[1])
            if not action:
                break

            nxt = get_next_position(current[0], current[1], action)

            if nxt == current:
                break

            path.append(nxt)
            current = nxt

        return jsonify({
            'success': True,
            'path': path,
            'found_treasure': current == TREASURE_POS
        })

    except Exception as e:
        print("❌ ERROR auto_run:", e)
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/reset', methods=['POST'])
def reset():
    return jsonify({
        'success': True,
        'position': [0, 0]
    })


@app.route('/api/state', methods=['GET'])
def state():
    return jsonify({
        'grid_size': GRID_SIZE,
        'treasure': TREASURE_POS,
        'walls': WALLS
    })

if __name__ == '__main__':
    app.run(debug=True)