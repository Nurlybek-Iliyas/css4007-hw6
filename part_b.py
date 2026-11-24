
import gymnasium as gym
import random
import csv
import os

# Фиксированная политика
fixed_policy = {
    0: 1, 1: 0, 2: 0, 3: 0,
    4: 1, 6: 1, 8: 2, 9: 1,
    10: 0, 13: 2, 14: 2
}

# Параметры эксперимента
batch_seeds = [11, 22, 33]
policies = ["random", "fixed"]
slippery_values = [False, True]
holes = [5, 7, 11, 12]
goal = 15
max_steps = 100

os.makedirs("outputs", exist_ok=True)

results = []
summaries = []
saved_success = False
saved_failure = False

# Запускаем эксперименты
for slippery in slippery_values:
    for policy_name in policies:
        for batch_seed in batch_seeds:

            env = gym.make(
                "FrozenLake-v1",
                map_name="4x4",
                is_slippery=slippery,
                render_mode="ansi"
            )

            for i in range(50):
                episode_seed = 1000 * batch_seed + i
                rng = random.Random(episode_seed)
                state, info = env.reset(seed=episode_seed)

                steps = 0
                total_reward = 0
                terminated = False
                truncated = False
                trace = env.render()

                while steps < max_steps:
                    if policy_name == "random":
                        action = rng.randrange(4)
                    else:
                        action = fixed_policy[state]

                    state, reward, terminated, truncated, info = env.step(action)

                    steps += 1
                    total_reward += reward
                    trace += env.render()

                    if terminated or truncated:
                        break

                # Определяем результат эпизода
                if terminated and state == goal:
                    outcome = "goal"
                elif terminated and state in holes:
                    outcome = "hole"
                else:
                    outcome = "time limit"

                results.append({
                    "condition": f"is_slippery={slippery}",
                    "batch_seed": batch_seed,
                    "episode_seed": episode_seed,
                    "policy": policy_name,
                    "steps": steps,
                    "total_reward": total_reward,
                    "outcome": outcome
                })

                # Сохраняем успешный и неуспешный примеры
                if outcome == "goal" and not saved_success:
                    with open("outputs/successful_trace.txt", "w", encoding="utf-8") as file:
                        file.write(trace)
                    saved_success = True

                if outcome != "goal" and not saved_failure:
                    with open("outputs/failed_trace.txt", "w", encoding="utf-8") as file:
                        file.write(trace)
                    saved_failure = True

            env.close()

            # Подсчитываем статистику набора
            batch_results = [
                row for row in results
                if row["condition"] == f"is_slippery={slippery}"
                and row["policy"] == policy_name
                and row["batch_seed"] == batch_seed
            ]

            goals = sum(row["outcome"] == "goal" for row in batch_results)
            hole_count = sum(row["outcome"] == "hole" for row in batch_results)
            time_limits = sum(row["outcome"] == "time limit" for row in batch_results)
            mean_steps = sum(row["steps"] for row in batch_results) / 50

            summaries.append({
                "condition": f"is_slippery={slippery}",
                "policy": policy_name,
                "batch_seed": batch_seed,
                "goals": goals,
                "holes": hole_count,
                "time_limits": time_limits,
                "success_percent": goals / 50 * 100,
                "mean_steps": mean_steps
            })

# Сохраняем результаты всех эпизодов
with open("outputs/part_b_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

# Сохраняем сводную статистику
with open("outputs/part_b_summary.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=summaries[0].keys())
    writer.writeheader()
    writer.writerows(summaries)

# Выводим сводку
print("PART B EXPERIMENT COMPLETED")
print("Total episodes:", len(results))
print()

for row in summaries:
    print(row)

print()
print("Files saved in outputs folder.")

