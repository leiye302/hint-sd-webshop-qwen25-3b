"""Real CPU search/purchase/reward and isolated child smoke; not Actor evaluation."""
import json
from .common import ROOT, atomic_json, task_rows, verify_sources
from .environment import Game, RemoteGame, native_parts, env_kwargs
from task_split import make_manifest, manifest, partition_goals, assert_allowed
from assets import verify
from fixed_validation import digest


def purchase(game):
    env = game.worker.env.unwrapped
    goal = env.server.user_sessions[env.session]['goal']
    product = env.server.product_item_dict[goal['asin']]
    actions = ['search[<q> '+product['query']+']']
    result = game.step('<think>Find target.</think><action>'+actions[-1]+'</action>')
    for _ in range(100):
        available = game.worker.get_available_actions()['clickables']
        if goal['asin'].lower() in available:
            break
        if 'next >' not in available:
            raise RuntimeError('Known target not reachable through native search')
        actions.append('click[next >]')
        game.step('<think>Next page.</think><action>'+actions[-1]+'</action>')
    actions.append('click['+goal['asin']+']')
    game.step('<think>Inspect matching product.</think><action>'+actions[-1]+'</action>')
    for value in goal['goal_options'].values():
        actions.append('click['+value+']')
        game.step('<think>Select required option.</think><action>'+actions[-1]+'</action>')
    actions.append('click[buy now]')
    result = game.step('<think>Required product and options match.</think><action>click[buy now]</action>')
    if result['reward'] != 10. or not result['won'] or not result['terminated'] or result['task_score'] != 1.:
        raise RuntimeError('Native exact-goal purchase did not produce the unchanged reward')
    return dict(result, actions=actions)


def main():
    verify_sources()
    receipt = verify()
    Worker,_,_,_ = native_parts()
    canonical = Worker(1000,env_kwargs())
    try:
        if make_manifest(canonical.env.unwrapped.server.goals) != manifest():
            raise RuntimeError('Cannot reconstruct shared140 split from native goals')
    finally:
        canonical.close()
    from .setup import prepare
    prepare()
    training = json.loads((ROOT/'data/train_tasks.json').read_text())
    rows = task_rows()
    evidence = []
    for row in [training[0],training[-1],rows[0],rows[-1]]:
        game = Game(row,15)
        try:
            initial = game.state()
            if initial['task_id'] != row['task_id'] or '<think>' not in initial['observation']:
                raise RuntimeError('Wrong actual task or native prompt')
            audit = game.worker.get_split_audit()
            if audit['opposite_split_overlap']:
                raise RuntimeError('Task leakage')
            if row['split']=='validation':
                result = purchase(game)
                game.worker._current_id()  # Includes native automatic terminal reset.
                evidence.append({'task_id':row['task_id'],'split':row['split'],
                    'initial_observation_sha256':digest(initial['observation']),
                    'reward':result['reward'],'task_score':result['task_score'],'actions':result['actions']})
            else:
                evidence.append(dict(audit,task_id=row['task_id']))
        finally:
            game.close()
    isolated = RemoteGame(rows[0],15)
    try:
        if isolated.initial['task_id'] != rows[0]['task_id']:
            raise RuntimeError('Spawned environment task differs')
        state = isolated.step('<think>Start search.</think><action>search[shoes]</action>')
        if not state['action_valid'] or not state['observation']:
            raise RuntimeError('Spawned environment cannot execute a native action')
    finally:
        isolated.close()
    atomic_json(ROOT/'results/environment_smoke.json',{'status':'passed',
        'not_actor_performance':True, 'fixed140_rebuilt':True, 'isolated_cpu_child_passed':True,
        'corpus_sha256':receipt['corpus_sha256'], 'cases':evidence})
    print('PASS: real native WebShop prompt/search/buy/reward, fixed140 exclusion and isolated CPU child.')


if __name__=='__main__':
    main()
