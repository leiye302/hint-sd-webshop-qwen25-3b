"""Synchronous inference replicas on separately allocated GPUs; public vLLM LoRA API, no private weight-loader hooks."""
import multiprocessing
import os
import shutil
import traceback
from pathlib import Path
from types import SimpleNamespace


def _worker(connection, model_path, gpu, max_model_len, utilization, seed):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '1'
    os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        os.environ.pop(name, None)
    llm = None
    try:
        from vllm import LLM
        from vllm.lora.request import LoRARequest
        llm = LLM(model=model_path, dtype='bfloat16', tensor_parallel_size=1,
                  max_model_len=max_model_len, gpu_memory_utilization=utilization,
                  enable_lora=True, max_lora_rank=32, max_loras=2, max_cpu_loras=2,
                  seed=seed, enforce_eager=True)
        connection.send({'ok': True})
        while True:
            request = connection.recv()
            if request['operation'] == 'close':
                break
            if request['operation'] != 'generate':
                raise ValueError('Unknown inference operation')
            adapter = request.get('adapter')
            lora = LoRARequest(adapter['name'], adapter['id'], adapter['path']) if adapter else None
            outputs = llm.generate(request['prompts'], sampling_params=request['sampling_params'],
                                   lora_request=lora, use_tqdm=False)
            packed = [{'prompt_token_ids': list(output.prompt_token_ids), 'outputs': [
                {'text': sample.text, 'token_ids': list(sample.token_ids), 'finish_reason': sample.finish_reason}
                for sample in output.outputs]} for output in outputs]
            connection.send({'ok': True, 'outputs': packed})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send({'ok': False, 'error': traceback.format_exc()})
    finally:
        if llm is not None:
            try:
                llm.llm_engine.engine_core.shutdown()
            except (AttributeError, RuntimeError):
                pass
        connection.close()


class RemoteEngine:
    def __init__(self, model_path, gpu, max_model_len, utilization, seed):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_worker, args=(child, str(model_path), str(gpu),
                                                            max_model_len, utilization, seed))
        self.adapter = None
        self.process.start()
        child.close()
        try:
            self._receive()
        except BaseException:
            self.close()
            raise

    def _receive(self):
        if not self.connection.poll(1800):
            raise TimeoutError('Inference worker did not respond within 30 minutes; inspect its log')
        result = self.connection.recv()
        if not result['ok']:
            raise RuntimeError('Inference worker failed:\n' + result['error'])
        return result

    def generate(self, prompts, sampling_params, use_tqdm=False):
        del use_tqdm
        self.connection.send({'operation': 'generate', 'prompts': prompts,
                              'sampling_params': sampling_params, 'adapter': self.adapter})
        packed = self._receive()['outputs']
        return [SimpleNamespace(prompt_token_ids=item['prompt_token_ids'], outputs=[
            SimpleNamespace(**sample) for sample in item['outputs']]) for item in packed]

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send({'operation': 'close'})
            except (EOFError, BrokenPipeError):
                pass
            self.process.join(30)
            if self.process.is_alive():
                self.process.terminate()  # only this job's owned inference process
                self.process.join(10)
        self.connection.close()


class AdapterPublisher:
    """Fresh immutable LoRA IDs on every native student/teacher synchronization."""
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.counter = 0
        self.paths = []

    def publish(self, trainer, adapter_name):
        self.counter += 1
        path = self.root / f'{adapter_name}-{self.counter:08d}'
        trainer._save_adapter_to_dir(adapter_name, str(path))  # original PEFT state exporter
        self.paths.append(path)
        while len(self.paths) > 2:
            previous = self.paths.pop(0).resolve()
            if previous.parent != self.root.resolve():
                raise ValueError('Refusing to remove an adapter outside this job')
            shutil.rmtree(previous)  # ephemeral inference copies, never training checkpoints
        return {'name': f'{adapter_name}-{self.counter}', 'id': self.counter, 'path': str(path.resolve())}

class RemotePool:
    """Three TP1 replicas; same request parameters, ordered outputs, no asynchronous stale policy."""
    def __init__(self, model_path, gpus, max_model_len, utilization, seed):
        self.engines = []
        self.adapter = None
        try:
            for index, gpu in enumerate(gpus):
                self.engines.append(RemoteEngine(model_path, gpu, max_model_len, utilization, seed+index))
        except BaseException:
            self.close()
            raise

    def generate(self, prompts, sampling_params, use_tqdm=False):
        from concurrent.futures import ThreadPoolExecutor
        if not prompts:
            return []
        outputs = [None]*len(prompts)
        assignments = [list(range(i, len(prompts), len(self.engines))) for i in range(len(self.engines))]
        with ThreadPoolExecutor(max_workers=len(self.engines)) as pool:
            futures = []
            for engine, indices in zip(self.engines, assignments, strict=True):
                if indices:
                    engine.adapter = self.adapter
                    params = [sampling_params[i] for i in indices] if isinstance(sampling_params, list) else sampling_params
                    futures.append((indices, pool.submit(engine.generate, [prompts[i] for i in indices], params, use_tqdm)))
            for indices, future in futures:
                returned = future.result()
                if len(returned) != len(indices):
                    raise RuntimeError('Inference replica output count differs')
                for index, result in zip(indices, returned, strict=True):
                    outputs[index] = result
        return outputs

    def close(self):
        for engine in self.engines:
            engine.close()
        self.engines.clear()
