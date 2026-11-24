
import random

# Создаём основную карту
grid = [
    [".", "M", ".", "G"],
    [".", "#", "I", "H"],
    ["S", ".", ".", "."]
]

# Задаём награды за клетки
rewards = {
    "S": -1,
    ".": -1,
    "M": -3,
    "I": -1,
    "G": 10,
    "H": -10
}

actions = ["U", "R", "D", "L"]

moves = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1)
}

# Задаём боковые направления для льда
perpendicular = {
    "U": ["L", "R"],
    "R": ["U", "D"],
    "D": ["L", "R"],
    "L": ["U", "D"]
}

gamma = 0.9
theta = 0.000001
max_iterations = 10000

rows = len(grid)
cols = len(grid[0])

states = [
    (r, c)
    for r in range(rows)
    for c in range(cols)
    if grid[r][c] != "#"
]

terminals = [
    (r, c)
    for r, c in states
    if grid[r][c] in ["G", "H"]
]


# Получаем возможные переходы
def get_transitions(state, action):
    r, c = state

    if state in terminals:
        return [(1.0, state, 0, True)]

    if grid[r][c] == "I":
        directions = [
            (0.8, action),
            (0.1, perpendicular[action][0]),
            (0.1, perpendicular[action][1])
        ]
    else:
        directions = [(1.0, action)]

    transitions = []

    for probability, direction in directions:
        dr, dc = moves[direction]
        nr = r + dr
        nc = c + dc

        # Проверяем границы и стены
        if (
            nr < 0 or nr >= rows
            or nc < 0 or nc >= cols
            or grid[nr][nc] == "#"
        ):
            nr, nc = r, c

        next_state = (nr, nc)
        reward = rewards[grid[nr][nc]]
        done = next_state in terminals

        transitions.append(
            (probability, next_state, reward, done)
        )

    return transitions


# Начальные значения состояний
V = {state: 0.0 for state in states}

V1 = None
V2 = None

# Выполняем Value Iteration
for iteration in range(max_iterations):
    new_V = V.copy()
    delta = 0

    for state in states:
        if state in terminals:
            new_V[state] = 0.0
            continue

        action_values = []

        for action in actions:
            value = 0

            for probability, next_state, reward, done in get_transitions(
                state, action
            ):
                future_value = 0 if done else V[next_state]

                value += probability * (
                    reward + gamma * future_value
                )

            action_values.append(value)

        new_V[state] = max(action_values)
        delta = max(delta, abs(new_V[state] - V[state]))

    V = new_V

    if iteration == 0:
        V1 = V.copy()

    if iteration == 1:
        V2 = V.copy()

    if delta < theta:
        break

print("Final delta:", delta)

# Выводим первые две итерации
print("V1:")
for state in [(0, 1), (0, 2), (1, 2)]:
    print(state, round(V1[state], 4))

print("\nV2:")
for state in [(0, 1), (0, 2), (1, 2)]:
    print(state, round(V2[state], 4))


# Выводим итоговые значения
print("\nFinal Values:")
for r in range(rows):
    line = []

    for c in range(cols):
        if grid[r][c] == "#":
            line.append("#")
        else:
            line.append(round(V[(r, c)], 4))

    print(line)


# Находим оптимальную политику
policy = {}

for state in states:
    if state in terminals:
        continue

    action_values = []

    for action in actions:
        value = 0

        for probability, next_state, reward, done in get_transitions(
            state, action
        ):
            future_value = 0 if done else V[next_state]

            value += probability * (
                reward + gamma * future_value
            )

        action_values.append(value)

    best_action = actions[action_values.index(max(action_values))]
    policy[state] = best_action


# Выводим оптимальную политику
print("\nOptimal Policy:")
for r in range(rows):
    line = []

    for c in range(cols):
        state = (r, c)

        if grid[r][c] == "#":
            line.append("#")
        elif state in terminals:
            line.append(grid[r][c])
        else:
            line.append(policy[state])

    print(line)

print("\nIterations:", iteration + 1)


# Проверяем модель
assert abs(sum(
    item[0] for item in get_transitions((1, 2), "U")
) - 1.0) < 1e-9

assert get_transitions((2, 0), "L")[0][1] == (2, 0)
assert get_transitions((1, 0), "R")[0][1] == (1, 0)

assert get_transitions((0, 3), "U")[0][2] == 0
assert get_transitions((1, 3), "U")[0][2] == 0

print("\nAll model checks passed!")


# Проводим симуляции основной карты
print("\nBase Map Simulation:")

for batch_seed in [11, 22, 33]:
    goals = 0
    holes = 0
    timeouts = 0
    total_steps = 0
    total_return = 0

    for episode in range(50):
        random.seed(1000 * batch_seed + episode)

        state = (2, 0)
        episode_steps = 0
        episode_return = 0
        discount = 1.0

        for step in range(100):
            action = policy[state]
            transitions = get_transitions(state, action)
            probabilities = [item[0] for item in transitions]

            chosen = random.choices(
                transitions,
                weights=probabilities
            )[0]

            probability, next_state, reward, done = chosen

            episode_return += discount * reward
            discount *= gamma

            state = next_state
            episode_steps += 1

            if done:
                if grid[state[0]][state[1]] == "G":
                    goals += 1
                elif grid[state[0]][state[1]] == "H":
                    holes += 1
                break
        else:
            timeouts += 1

        total_steps += episode_steps
        total_return += episode_return

    print(
        "Batch:", batch_seed,
        "Goals:", goals,
        "Holes:", holes,
        "Time limits:", timeouts,
        "Mean steps:", round(total_steps / 50, 2),
        "Mean discounted return:", round(total_return / 50, 4)
    )