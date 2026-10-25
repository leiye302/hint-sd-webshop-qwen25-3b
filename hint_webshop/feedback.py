"""Necessary domain substitution in the official hindsight instruction; same JSON contract."""


def feedback_messages(trajectory, max_steps=3):
    return [
        {'role': 'system', 'content':
         'You analyze failed WebShop shopping-agent trajectories. '
         f'Identify up to {max_steps} problematic steps where the agent made mistakes '
         f'(between 1 and {max_steps} steps total, ordered earliest first). '
         'Examples include a wrong product, missed requested attribute or option, '
         'misreading a page/search result, premature purchase, or an invalid tool argument. '
         'Do not call ordinary exploration a mistake merely because it did not immediately succeed. '
         'For EACH problematic step, write a distinct correction in less than three sentences '
         'that targets only that step\'s mistake. Output valid JSON only, no other text.'},
        {'role': 'user', 'content':
         'Review this failed WebShop attempt and identify the problematic steps. '
         'Return one (step, feedback) pair per problematic step.\n\n' + trajectory +
         '\n\nOutput format (JSON only, no other text):\n'
         '{"failures": [{"step": <1-indexed step number>, "feedback": "<correction for this step>"}, ...]}'}]
