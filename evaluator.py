from databench_eval import Evaluator


def evaluate_answer(predicted, expected, answer_type):
    evaluator = Evaluator.__new__(Evaluator)

    is_correct = evaluator.default_compare(
        predicted,
        expected,
        answer_type
    )

    return is_correct