"""Global, price-invariant WebShop task identities and disjoint train/val pools."""
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import re
from common import ROOT, load, sha

PRICE_SUFFIX = re.compile(r', and price lower than [0-9]+(?:[.][0-9]+)? dollars\s*$', re.I)
COUNT = 140
CATEGORIES = ('beauty', 'electronics', 'fashion', 'garden', 'grocery')
PER_CATEGORY = 28
SELECTION_SEED = 2026


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def normalize(value):
    return ' '.join(str(value).lower().split())


def task_identity(goal):
    # Do not use per-worker goal indices, ASIN, or randomized price thresholds.
    # Identical visible objectives under different products stay in one family.
    instruction = PRICE_SUFFIX.sub('', goal['instruction_text'])
    options = goal['goal_options']
    if isinstance(options, dict):
        options = sorted((normalize(k), normalize(v)) for k, v in options.items())
    else:
        options = sorted(normalize(v) for v in options)
    return digest({'instruction_without_price': normalize(instruction),
                   'attributes': sorted(normalize(a) for a in goal['attributes']),
                   'options': options})


def select_validation(canonical_goals):
    """Balanced public categories, varied constraints, no repeated target product.

    Selection uses metadata only, never model output, reward or difficulty scores.
    The price-independent key is needed because native workers randomize prices.
    Training entries are not otherwise deduplicated, rewritten or reweighted.
    """
    representatives = {}
    for goal in canonical_goals:
        key = task_identity(goal)
        previous = representatives.get(key)
        if previous is None or (str(goal['asin']), str(goal['name'])) < (str(previous['asin']), str(previous['name'])):
            representatives[key] = goal
    selected, products = [], set()
    for category in CATEGORIES:
        candidates = {key: goal for key, goal in representatives.items()
                      if normalize(goal['category']) == category}
        buckets, queries, paths = Counter(), Counter(), Counter()
        for _ in range(PER_CATEGORY):
            available = [(key, goal) for key, goal in candidates.items() if goal['asin'] not in products]
            if not available:
                raise RuntimeError('Not enough distinct products for balanced validation: ' + category)
            def priority(pair):
                key, goal = pair
                bucket = (len(goal['attributes']), len(goal['goal_options']))
                return (buckets[bucket], queries[normalize(goal['query'])],
                        paths[normalize(goal['product_category'])],
                        digest([SELECTION_SEED, key]))
            key, goal = min(available, key=priority)
            selected.append((key, goal))
            products.add(goal['asin'])
            buckets[(len(goal['attributes']), len(goal['goal_options']))] += 1
            queries[normalize(goal['query'])] += 1
            paths[normalize(goal['product_category'])] += 1
    return selected


def make_manifest(canonical_goals):
    ids = [task_identity(g) for g in canonical_goals]
    unique = list(dict.fromkeys(ids))
    selected = select_validation(canonical_goals)
    validation = [key for key, _ in selected]
    if len(validation) != COUNT or len(set(validation)) != COUNT:
        raise RuntimeError('Expected exactly140 distinct validation objectives')
    training = sorted(set(unique) - set(validation))
    validation_set = set(validation)
    return {
        'schema': 1, 'protocol': 'webshop-balanced140-disjoint-v1',
        'canonical_catalog_seed': 1000,
        'selection_seed': SELECTION_SEED,
        'selection': '28 per native category; unique target products; balance attribute/option counts, query and product category; hash tie-break',
        'identity': 'normalized instruction minus generated price suffix + attributes + goal_options; no ASIN',
        'upstream_commit': load('upstream.json')['commit'],
        'assets_lock_sha256': sha(ROOT / 'configs/assets.json'),
        'raw_goal_entries': len(ids), 'unique_task_families': len(unique),
        'training_goal_entries': sum(i not in validation_set for i in ids),
        'validation_goal_entries_before_dedup': sum(i in validation_set for i in ids),
        'catalog_task_ids_sha256': digest(sorted(unique)),
        'training_task_families': len(training),
        'validation_task_ids': validation,
        'validation_tasks': [{'task_id': key, 'asin': goal['asin'], 'category': goal['category'],
            'query': goal['query'], 'product_category': goal['product_category'],
            'attribute_count': len(goal['attributes']), 'option_count': len(goal['goal_options'])}
            for key, goal in selected],
        'diversity': {'category_counts': dict(sorted(Counter(goal['category'] for _, goal in selected).items())),
            'distinct_products': len({goal['asin'] for _, goal in selected}),
            'distinct_queries': len({normalize(goal['query']) for _, goal in selected}),
            'distinct_product_categories': len({normalize(goal['product_category']) for _, goal in selected}),
            'attribute_counts': dict(sorted(Counter(str(len(goal['attributes'])) for _, goal in selected).items())),
            'option_counts': dict(sorted(Counter(str(len(goal['goal_options'])) for _, goal in selected).items()))},
    }


@lru_cache(maxsize=1)
def manifest():
    value = load('task_split.json')
    val = value['validation_task_ids']
    if (value['schema'] != 1 or value['protocol'] != 'webshop-balanced140-disjoint-v1'
            or len(val) != COUNT or len(set(val)) != COUNT
            or value['training_task_families'] + len(val) != value['unique_task_families']
            or [row['task_id'] for row in value['validation_tasks']] != val
            or len({row['asin'] for row in value['validation_tasks']}) != COUNT
            or value['diversity']['category_counts'] != {key: PER_CATEGORY for key in CATEGORIES}
            or value['upstream_commit'] != load('upstream.json')['commit']
            or value['assets_lock_sha256'] != sha(ROOT / 'configs/assets.json')
            or sha(ROOT / 'configs/task_split.json') != load('upstream.json')['task_split_sha256']):
        raise RuntimeError('Invalid or overlapping WebShop train/validation manifest')
    return value


def partition_goals(goals, is_train):
    value = manifest()
    groups = defaultdict(list)
    for goal in goals:
        groups[task_identity(goal)].append(goal)
    validation = validation_ids()
    if (digest(sorted(groups)) != value['catalog_task_ids_sha256']
            or len(goals) != value['raw_goal_entries']):
        raise RuntimeError('Native goal catalog differs from the pinned global split')
    if is_train:
        # Preserve native shuffled order, entries and weights within the allowed pool.
        selected = [g for g in goals if task_identity(g) not in validation]
        if len(selected) != value['training_goal_entries']:
            raise RuntimeError('Training goal count differs from the manifest')
    else:
        # A stable representative prevents duplicate product versions of one task.
        selected = [min(groups[key], key=lambda g: (str(g['asin']).lower(), str(g['name'])))
                    for key in value['validation_task_ids']]
    if (not selected or any((task_identity(g) in validation) == is_train for g in selected)):
        raise RuntimeError('A forbidden task entered an environment pool')
    return selected


def assert_allowed(goal, is_train):
    key = task_identity(goal)
    if (key in validation_ids()) == is_train:
        raise RuntimeError('Training/validation task leakage detected')
    return key


@lru_cache(maxsize=1)
def validation_ids():
    return frozenset(manifest()['validation_task_ids'])
