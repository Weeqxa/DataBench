from databench_eval import Evaluator

# Створюємо оцінювач без завантаження датасету
evaluator = Evaluator.__new__(Evaluator)

# Наша відповідь
predicted = "19"

# Правильна відповідь DataBench
expected = "19.0"

# Тип відповіді
answer_type = "number"

# Перевіряємо через офіційну функцію
is_correct = evaluator.default_compare(
    predicted,
    expected,
    answer_type
)

print("Predicted:", predicted)
print("Expected:", expected)
print("Correct:", is_correct)