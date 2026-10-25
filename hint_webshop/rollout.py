"""Native WebShop interaction, retaining exact sampled prompt/completion token IDs."""
import copy
from concurrent.futures import ThreadPoolExecutor
from .environment import Client, policy_messages


def run_rollouts(rows, trainer, *, rollout_n, temperature):
    from env.integration.bfcl.env_service_backend import BfclEpisode, BfclStep
    from src.train.bfcl_rollout import BfclRolloutBatch, TokenizedBfclTrajectory
    config = trainer.recipe
    client = Client(trainer.environment_url)
    jobs = [(copy.deepcopy(row), index) for row in rows for index in range(rollout_n)]
    created = []
    episodes = []
    pool = ThreadPoolExecutor(max_workers=config['max_env_workers'])
    try:
        # Keep create receipts even when a later creation fails, so finally releases all instances.
        futures = [pool.submit(client.call, 'create', task=row) for row, _ in jobs]
        creation_error = None
        for future in futures:
            try:
                created.append(future.result())
            except BaseException as error:
                created.append(None)
                creation_error = creation_error or error
        if creation_error is not None:
            raise creation_error
        states = list(created)
        for (row, rollout_index), state in zip(jobs, states, strict=True):
            messages = policy_messages(state['observation'], state['admissible_commands'], state['task'], [],
                                       config['history_length'])
            episodes.append(BfclEpisode(task_id=row['task_id'], dataset_name='webshop',
                                        instance_id=state['instance_id'], initial_state=messages,
                                        run_metadata={'rollout_index': rollout_index, 'seed': row['seed'], 'split': row['split']}))
        active = list(range(len(episodes)))
        trainer._ensure_vllm_adapter_weights(trainer.student_adapter_name)
        for step_index in range(config['max_env_steps']):
            if not active:
                break
            prompts, messages_list, valid_indices = [], [], []
            for index in active:
                state = states[index]
                messages = policy_messages(state['observation'], state['admissible_commands'], state['task'],
                                           state['history'], config['history_length'])
                prompt = trainer._apply_chat_template(messages, add_generation_prompt=True)
                tokens = trainer._encode_text(prompt)
                if len(tokens) > config['max_prompt_tokens']:
                    # SDAR uses left prompt truncation. Store the exact retained IDs used for generation.
                    tokens = tokens[-config['max_prompt_tokens']:]
                prompts.append({'prompt_token_ids': tokens})
                messages_list.append(messages)
                valid_indices.append(index)
            outputs = trainer._sample_model_outputs(prompts, num_generations=1, temperature=temperature)
            if len(outputs) != len(valid_indices):
                raise RuntimeError('Rollout output count differs from environment count')
            future_steps = [pool.submit(client.call, 'step', instance_id=episodes[index].instance_id,
                                        response=output.outputs[0].text)
                            for index, output in zip(valid_indices, outputs, strict=True)]
            next_active = []
            step_error = None
            for index, messages, output, future in zip(valid_indices, messages_list, outputs, future_steps, strict=True):
                try:
                    state = future.result()
                except BaseException as error:
                    step_error = step_error or error
                    continue
                states[index] = state
                sample = output.outputs[0]
                episode = episodes[index]
                metadata = {'prompt_token_ids': list(output.prompt_token_ids),
                            'completion_token_ids': list(sample.token_ids),
                            'student_messages': messages, 'finish_reason': sample.finish_reason,
                            'command': state['command'], 'action_valid': state['action_valid'],
                            'executable': state['executable'], 'task_score': state.get('task_score', 0.)}
                episode.steps.append(BfclStep(index=step_index,
                    action={'role': 'assistant', 'content': sample.text},
                    state=[{'role': 'user', 'content': state.get('raw_observation', state['observation'])}], reward=state['reward'],
                    is_terminated=state['terminated'], info={'won': state['won']}, agent_metadata=metadata))
                episode.final_reward, episode.success, episode.terminated = state['reward'], state['won'], state['terminated']
                if not state['terminated']:
                    next_active.append(index)
            if step_error is not None:
                raise step_error
            active = next_active
        tokenized = []
        for episode in episodes:
            first = episode.steps[0].agent_metadata
            # Native multi-step HINT-SD reads exact per-step IDs above. These zero-mask placeholders
            # serve only its passthrough path for success/no-detection episodes.
            tokenized.append(TokenizedBfclTrajectory(
                prompt_messages=episode.initial_state, prompt_text='', completion_text='',
                prompt_ids=first['prompt_token_ids'], completion_ids=first['completion_token_ids'],
                completion_attention_mask=[1] * len(first['completion_token_ids']),
                completion_loss_mask=[0] * len(first['completion_token_ids']), step_token_ranges=[]))
        count = max(1, len(episodes))
        return BfclRolloutBatch(episodes=episodes, tokenized=tokenized,
            scores=[float(episode.success) for episode in episodes],
            feedback=[None if episode.success else episode.steps[-1].state[0]['content'] for episode in episodes],
            metrics={'webshop/success_rate': sum(episode.success for episode in episodes) / count,
                     'webshop/avg_steps': sum(len(episode.steps) for episode in episodes) / count})
    finally:
        # Join step requests before releasing games; never let an in-flight action race cleanup.
        pool.shutdown(wait=True)
        errors = []
        for item in created:
            if item is not None:
                try:
                    client.call('release', instance_id=item['instance_id'])
                except Exception as error:
                    errors.append(str(error))
        if not errors:
            for episode in episodes:
                episode.released = True
        if errors:
            raise RuntimeError('Failed to release owned WebShop instances: ' + '; '.join(errors))
