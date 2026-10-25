"""Aggregate actual Student evaluation outputs; never manufacture benchmark results."""
import csv
import json
import re
from .common import ROOT


def protocol(text):
    reasoning = re.search(r'<think>(.*?)</think>', text, re.S)
    clean = text.replace('<|im_end|>', '').replace('<|endoftext|>', '')
    match = re.fullmatch(r'\s*<think>(.*?)</think>\s*<action>(.*?)</action>\s*', clean, re.S)
    return {'reasoning': bool(reasoning and reasoning[1].strip()),
            'strict': bool(match and match[1].strip() and match[2].strip()
                           and clean.count('<think>') == clean.count('</think>') == 1
                           and clean.count('<action>') == clean.count('</action>') == 1)}


def summarize(episodes):
    steps = [step for episode in episodes for step in episode.steps]
    count = max(1, len(steps))
    flags = [protocol(step.action['content']) for step in steps]
    successes = sum(episode.success for episode in episodes)
    return {'tasks': len(episodes), 'successes': successes, 'success_rate': successes / max(1, len(episodes)),
            'task_score': sum(e.steps[-1].agent_metadata.get('task_score', 0.) for e in episodes)/max(1,len(episodes)),
            'decisions': len(steps), 'reasoning_rate': sum(flag['reasoning'] for flag in flags) / count,
            'strict_protocol_rate': sum(flag['strict'] for flag in flags) / count,
            'native_action_valid_rate': sum(step.agent_metadata['action_valid'] for step in steps) / count,
            'current_executable_rate': sum(step.agent_metadata['executable'] for step in steps) / count,
            'length_truncation_rate': sum(step.agent_metadata['finish_reason'] == 'length' for step in steps) / count}


def main():
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    rows = []
    for path in sorted((ROOT / 'runs').glob('hint_sd_*/evaluation/update_*/summary.json')):
        result = json.loads(path.read_text())
        rows.append({'run': path.parents[2].name, 'update': result['update'],
                     'successes': result['successes'], 'tasks': result['tasks'],
                     'success_rate': result['success_rate'], 'reasoning_rate': result['reasoning_rate']})
    with (output / 'fixed140.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['run', 'update', 'successes', 'tasks', 'success_rate', 'reasoning_rate'])
        writer.writeheader()
        writer.writerows(rows)
    print('Wrote actual evaluation rows:', len(rows))


if __name__ == '__main__':
    main()
