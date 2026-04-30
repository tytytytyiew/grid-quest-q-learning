# Grid Quest — Q-Learning Agent

## Описание

Проект реализует алгоритм Q-learning, в котором агент обучается находить оптимальный путь в сетке (grid environment), получая награды и штрафы за действие.

## Структура проекта

```
grid-quest-q-learning/
├── app.py
├── train.py
├── q_model.pkl
├── requirements.txt
├── templates/
├── static/
```

## Установка

```bash
git clone https://github.com/tytytytyiew/grid-quest-q-learning.git
cd grid-quest-q-learning
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Обучение

```bash
python train.py
```

## Запуск

```bash
python app.py
```

Открыть в браузере:
http://127.0.0.1:5000/

## Алгоритм

Обновление Q-значений:

```
Q(s, a) = Q(s, a) + α [r + γ max Q(s', a') - Q(s, a)]
```

## Зависимости

См. requirements.txt
