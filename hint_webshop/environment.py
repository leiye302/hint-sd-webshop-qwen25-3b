"""CPU-only process-isolated WebShop service; unedited native prompt/parser/reward."""
import argparse
import ast
import copy
import importlib.util
import json
import multiprocessing
import os
import signal
import sys
import threading
import traceback
import uuid
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from .common import ROOT, SDAR, assets, load, verify_sources
sys.path.insert(0, str(ROOT / 'src'))


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def native_class(path, name, namespace, methods=None):
    """Execute unchanged selected native ASTs without importing Ray's training stack."""
    node = copy.deepcopy(next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                              if isinstance(n, ast.ClassDef) and n.name == name))
    node.bases = [ast.Name(id='object', ctx=ast.Load())]
    if methods is not None:
        node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), node], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), namespace)
    return namespace[name]


@lru_cache(maxsize=1)
def native_parts():
    import gym
    import numpy as np
    path = SDAR / 'agent_system/environments/env_package/webshop/envs.py'
    worker = native_class(path, 'WebshopWorker', {'gym':gym, 'np':np, '__file__':str(path)})
    prompts = module_at('hint_webshop_native_prompts', SDAR/'agent_system/environments/prompts/webshop.py')
    projection = module_at('hint_webshop_native_projection',
        path.parent/'projection.py').webshop_projection
    memory = native_class(SDAR/'agent_system/memory/memory.py', 'SimpleMemory', {})
    formatter = native_class(SDAR/'agent_system/environments/env_manager.py',
        'WebshopEnvironmentManager', {
            'WEBSHOP_TEMPLATE':prompts.WEBSHOP_TEMPLATE,
            'WEBSHOP_TEMPLATE_NO_HIS':prompts.WEBSHOP_TEMPLATE_NO_HIS},
        {'extract_task','format_obs','format_avail_actions','build_text_obs'})
    return worker, formatter, memory, projection


def env_kwargs():
    return {'observation_mode':'text', 'num_products':None, 'human_goals':False,
            'file_path':str(assets()/'data/items_shuffle_1000.json'),
            'attr_path':str(assets()/'data/items_ins_v2_1000.json')}


def policy_messages(observation, admissible, task, history, history_length=2):
    # Observation is already rendered by the ORIGINAL WebshopEnvironmentManager.
    # No second chat/history formatting is applied in the training process.
    del admissible, task, history, history_length
    return [{'role':'user', 'content':observation}]


class Game:
    """One instance per CPU child. Never share native RNG/catalog/session between threads."""
    def __init__(self, row, max_steps):
        from webshop_worker import DisjointWebshopWorker
        if row.get('split') not in ('train', 'validation'):
            raise ValueError('An explicit allowed task split is required')
        _, Formatter, Memory, self.projection = native_parts()
        self.worker = DisjointWebshopWorker(int(row['seed']), env_kwargs(), row['split']=='train')
        try:
            raw, self.info = self.worker.reset(int(row['goal_index']))
            if self.info['task_id'] != row['task_id']:
                raise ValueError('Actual environment task differs from requested content ID')
            self.manager = Formatter()
            self.manager.tasks = self.manager.extract_task([raw])
            self.manager.config = SimpleNamespace(env=SimpleNamespace(history_length=load('experiment.json')['history_length']))
            self.manager.memory = Memory()
            self.manager.memory.reset(batch_size=1)
            self.formatted = self.manager.format_obs([raw])
            self.raw_observation = raw
            self.observation = self.manager.build_text_obs(self.formatted, [self.info], init=True)[0]
            self.task_id = row['task_id']
            self.task = self.manager.tasks[0]
        except BaseException:
            self.worker.close()
            raise

    def state(self):
        available = self.manager.format_avail_actions(self.info['available_actions'])
        return {'observation':self.observation, 'raw_observation':self.raw_observation,
                'admissible_commands':available, 'task':self.task, 'history':[],
                'task_id':self.task_id}

    def step(self, response):
        import re
        commands, valid = self.projection([response])
        available = self.info['available_actions']
        command = commands[0]
        click = re.fullmatch(r'click\[(.*)\]', command)
        executable = bool((command.startswith('search[') and available['has_search_bar']) or
                          (click and click[1].lower() in [str(x).lower() for x in available['clickables']]))
        raw, reward, done, self.info = self.worker.step(command)
        self.manager.memory.store({'text_obs':self.formatted, 'action':commands})
        self.formatted = self.manager.format_obs([raw])
        self.raw_observation = raw
        self.observation = self.manager.build_text_obs(self.formatted, [self.info])[0]
        if self.info['task_id'] != self.task_id:
            raise RuntimeError('Environment changed the submitted task')
        return dict(self.state(), command=command, action_valid=bool(valid[0]),
                    executable=executable, won=bool(self.info['won']), terminated=bool(done),
                    reward=float(reward), task_score=float(self.info['task_score']))

    def close(self):
        self.worker.close()


def _game_worker(connection, row, max_steps):
    game = None
    try:
        game = Game(row, max_steps)
        connection.send({'ok':True, 'state':game.state()})
        while True:
            item = connection.recv()
            if item['operation'] == 'close':
                break
            if item['operation'] != 'step':
                raise ValueError('Unknown child operation')
            connection.send({'ok':True, 'state':game.step(item['response'])})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send({'ok':False, 'error':traceback.format_exc()})
    finally:
        if game is not None:
            game.close()
        connection.close()


class RemoteGame:
    def __init__(self, row, max_steps):
        context = multiprocessing.get_context('spawn')
        self.connection, child = context.Pipe()
        self.process = context.Process(target=_game_worker, args=(child, row, max_steps))
        self.lock = threading.Lock()
        self.process.start()
        child.close()
        try:
            self.initial = self.receive()
        except BaseException:
            self.close()
            raise

    def receive(self):
        if not self.connection.poll(300):
            raise TimeoutError('Owned CPU WebShop child timed out; inspect environment.log')
        value = self.connection.recv()
        if not value['ok']:
            raise RuntimeError('Native WebShop child failed:\n'+value['error'])
        return value['state']

    def step(self, response):
        self.connection.send({'operation':'step', 'response':response})
        return self.receive()

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send({'operation':'close'})
            except (EOFError, BrokenPipeError):
                pass
            self.process.join(20)
            if self.process.is_alive():
                self.process.terminate()  # only this job's owned process
                self.process.join(10)
        self.connection.close()


class Service:
    def __init__(self, max_steps=15):
        self.games = {}
        self.lock = threading.Lock()
        self.max_steps = max_steps

    def call(self, operation, value):
        if operation == 'health':
            return {'ready':True, 'pid':os.getpid()}
        if operation == 'create':
            game = RemoteGame(value['task'], self.max_steps)
            identity = uuid.uuid4().hex
            with self.lock:
                self.games[identity] = game
            return dict(game.initial, instance_id=identity)
        identity = value['instance_id']
        with self.lock:
            game = self.games.get(identity)
        if operation == 'release':
            if game is not None:
                with game.lock:
                    game.close()
                with self.lock:
                    self.games.pop(identity, None)
            return {'released':True}
        if game is None:
            raise ValueError('Unknown/released environment instance')
        if operation == 'step':
            with game.lock:
                return game.step(str(value['response']))
        raise ValueError('Unknown operation')

    def close(self):
        for identity in list(self.games):
            self.call('release', {'instance_id':identity})


class Client:
    def __init__(self, url):
        if not url.startswith('http://127.0.0.1:'):
            raise ValueError('Environment service must use loopback')
        self.url = url.rstrip('/')

    def call(self, operation, *, timeout=300, **payload):
        import urllib.request
        request = urllib.request.Request(self.url+'/'+operation,
            data=json.dumps(payload).encode(), headers={'Content-Type':'application/json'}, method='POST')
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18760)
    args = parser.parse_args()
    verify_sources()
    from assets import verify
    verify()
    service = Service(load('experiment.json')['max_env_steps'])

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                length = int(self.headers.get('Content-Length', 0))
                if not 0 <= length <= 4*1024**2:
                    raise ValueError('Invalid request size')
                result = service.call(self.path.lstrip('/'), json.loads(self.rfile.read(length) or b'{}'))
                encoded = json.dumps(result).encode()
                self.send_response(200)
            except Exception as error:
                encoded = json.dumps({'error':type(error).__name__, 'message':str(error)}).encode()
                self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)
        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    def terminate(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    try:
        print('WebShop CPU service ready on loopback.', flush=True)
        server.serve_forever()
    finally:
        server.server_close()
        service.close()


if __name__ == '__main__':
    main()
