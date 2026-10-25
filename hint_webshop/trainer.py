"""Subclass the pinned trainer ONLY at environment, inference transport and evaluation boundaries."""
import json
import os
from pathlib import Path
from src.train.trainer import Trainer as OfficialTrainer
from src.train.appworld_feedback import parse_multi_step_failure_json
from src.train.bfcl_feedback import render_episode_for_detection
from .common import atomic_json, task_rows
from .engine import AdapterPublisher, RemotePool
from .checkpoint_io import VerifiedCheckpointMixin
from .feedback import feedback_messages
from .rollout import run_rollouts


class WebshopTrainer(VerifiedCheckpointMixin, OfficialTrainer):
    def __init__(self, *args, recipe, environment_url, **kwargs):
        self.recipe = recipe
        self.environment_url = environment_url
        super().__init__(*args, **kwargs)

    def _init_vllm_engine(self):
        self.publisher = AdapterPublisher(Path(self.args.output_dir) / 'inference_adapters')
        return RemotePool(self.model.config._name_or_path, os.environ['HINTSD_INFERENCE_GPUS'].split(','),
                            self.recipe['max_model_len'], self.recipe['vllm_gpu_memory_utilization'],
                            self.args.seed)

    def _sync_adapter_weights_to_vllm(self, adapter_name):
        # Equivalent active-adapter weights, without the upstream private V0 driver_worker access.
        self.llm.adapter = self.publisher.publish(self, adapter_name)

    def close(self):
        self.llm.close()

    def _prepare_inputs(self, generation_batch):
        if 'prompt_ids' in generation_batch:
            return generation_batch
        batch = run_rollouts(generation_batch['rows'], self, rollout_n=self.recipe['rollout_n'],
                             temperature=self.recipe['temperature'])
        output = Path(self.args.output_dir) / 'rollouts'
        output.mkdir(parents=True, exist_ok=True)
        with (output / 'train.jsonl').open('a', encoding='utf-8') as stream:
            for episode in batch.episodes:
                stream.write(json.dumps({'update_before': self.state.global_step, **episode.to_dict()}) + '\n')
        metrics = {**batch.metrics, 'rollout/mean_score': sum(batch.scores) / max(1, len(batch.scores))}
        # Selection, success skipping, selected completion masks and tensor construction are NATIVE.
        return self._prepare_bfcl_multi_step_inputs(batch, rollout_metrics=metrics)

    def _detect_bfcl_failure_steps_multi(self, episodes, env_feedback, *, max_steps=3):
        results = [None] * len(episodes)
        prompts, indices = [], []
        for index, episode in enumerate(episodes):
            if episode.success or not episode.steps:
                continue
            trajectory = render_episode_for_detection(episode, env_feedback[index],
                max_chars=self.recipe['feedback_max_trajectory_chars'])
            prompts.append(self._apply_chat_template(feedback_messages(trajectory, max_steps), True))
            indices.append(index)
        generated = self._sample_teacher_model_texts(prompts, max_new_tokens=self.recipe['feedback_max_tokens'])
        if len(generated) != len(indices):
            raise RuntimeError('Feedback output count differs from failed rollout count')
        audit = []
        for index, raw in zip(indices, generated, strict=True):
            parsed = parse_multi_step_failure_json(raw, len(episodes[index].steps), max_steps=max_steps)
            results[index] = parsed
            audit.append({'task_id': episodes[index].task_id, 'raw': raw, 'selected': parsed})
        with (Path(self.args.output_dir) / 'feedback.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'update_before': self.state.global_step, 'feedback': audit}) + '\n')
        return results

    def _build_bfcl_per_step_teacher_prompt(self, episode, failure_step_index, hint):
        # Use exactly the actual (possibly left-truncated) Student prefix, not a reconstructed
        # full trajectory. This special message is rendered by the adapter below, not by Qwen.
        return [{'role': '__exact_hindsight__', 'content': hint,
                 'prefix_ids': episode.steps[failure_step_index].agent_metadata['prompt_token_ids']}]

    def _apply_chat_template(self, messages, add_generation_prompt):
        if messages and messages[0]['role'] == '__exact_hindsight__':
            item = messages[0]
            # Qwen2.5 has no thinking toggle. Infer its native assistant header, never hand-tokenize tags.
            probe = [{'role': 'user', 'content': 'prefix-probe'}]
            closed = super()._apply_chat_template(probe, False)
            opened = super()._apply_chat_template(probe, True)
            if not opened.startswith(closed):
                raise ValueError('Unsupported non-prefix-stable chat template')
            header = opened[len(closed):]
            raw = self.processing_class.decode(item['prefix_ids'], skip_special_tokens=False)
            if not header or not raw.endswith(header):
                raise ValueError('Recorded Student prefix does not end at the native assistant boundary')
            feedback = [{'role': 'user', 'content': 'Consider the following feedback to solve this step:\n\n' + item['content']}]
            return raw[:-len(header)] + super()._apply_chat_template(feedback, add_generation_prompt)
        return super()._apply_chat_template(messages, add_generation_prompt)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix='eval'):
        del eval_dataset, ignore_keys
        rows = task_rows()[:getattr(self, 'evaluation_limit', 140)]
        episodes = []
        batch_size = self.recipe['eval_batch_size']
        for start in range(0, len(rows), batch_size):
            batch = run_rollouts(rows[start:start + batch_size], self, rollout_n=1,
                                 temperature=self.recipe['eval_temperature'])
            episodes.extend(batch.episodes)
        from .report import summarize
        summary = summarize(episodes)
        from fixed_validation import digest, record_identity, manifest
        from .common import sha, ROOT
        observations = [e.initial_state[0]['content'] for e in episodes]
        fixed = manifest()
        if [e.task_id for e in episodes] != [r['task_id'] for r in rows]:
            raise RuntimeError('Evaluation tasks/order changed')
        identity = {'protocol': fixed['protocol'],
            'manifest_sha256': sha(ROOT/'configs/fixed_validation140.json'),
            'task_split_sha256': sha(ROOT/'configs/task_split.json'),
            'task_ids': [r['task_id'] for r in rows],
            'selection_sha256': digest(fixed['selection'][:len(rows)]),
            'initial_observations_sha256': digest(observations)}
        record_identity(self.args.output_dir, identity)
        summary['fixed_validation_identity'] = identity
        summary.update({'update': self.state.global_step, 'task_ids': [row['task_id'] for row in rows],
                        'seeds': [row['seed'] for row in rows], 'temperature': self.recipe['eval_temperature'],
                        'max_steps': self.recipe['max_env_steps'], 'student_only': True})
        output = Path(self.args.output_dir) / 'evaluation' / f'update_{self.state.global_step:03d}'
        atomic_json(output / 'decisions.json', [episode.to_dict() for episode in episodes])
        atomic_json(output / 'summary.json', summary)
        metrics = {f'{metric_key_prefix}_success_rate': summary['success_rate'],
                   f'{metric_key_prefix}_successes': summary['successes'],
                   f'{metric_key_prefix}_tasks': summary['tasks']}
        self.log(metrics)
        return metrics
