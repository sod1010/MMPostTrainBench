"""Bounded operator-owned instance resources, shared by broker and CLI grading."""
import json
import re


DEFAULT_INSTANCE_RESOURCES = {'cpus': 8, 'memory_gib': 16, 'pids': 512}
LIMITS = {'cpus': (1, 256), 'memory_gib': (1, 1024), 'pids': (1, 65536)}
RESOURCE_ENV = 'MMSWE_INSTANCE_RESOURCES_JSON'


def instance_resources(value=None):
    value = DEFAULT_INSTANCE_RESOURCES if value is None else value
    if not isinstance(value, dict) or set(value) != set(LIMITS):
        raise ValueError('instance resources require cpus, memory_gib and pids')
    for key, (low, high) in LIMITS.items():
        if type(value[key]) is not int or not low <= value[key] <= high:
            raise ValueError(f'instance {key} must be an integer in {low}..{high}')
    return dict(value)


def resource_environment(value):
    return {RESOURCE_ENV: json.dumps(instance_resources(value), sort_keys=True)}


def resources_from_environment(env):
    aliases = {'MMSWE_DOCKER_PIDS', 'MMSWE_DOCKER_MEM'}
    if RESOURCE_ENV in env:
        if aliases.intersection(env):
            raise ValueError('conflicting instance resource JSON and legacy resource switches')
        return instance_resources(json.loads(env[RESOURCE_ENV]))
    result = instance_resources()
    if 'MMSWE_DOCKER_PIDS' in env:
        value = env['MMSWE_DOCKER_PIDS']
        if not re.fullmatch(r'[1-9][0-9]{0,4}', value):
            raise ValueError('MMSWE_DOCKER_PIDS must be a bounded positive integer')
        result['pids'] = int(value)
    if 'MMSWE_DOCKER_MEM' in env:
        value = env['MMSWE_DOCKER_MEM']
        if not re.fullmatch(r'[1-9][0-9]{0,3}[gG]', value):
            raise ValueError('MMSWE_DOCKER_MEM must be a whole GiB value, such as 32g')
        result['memory_gib'] = int(value[:-1])
    return instance_resources(result)


def resource_args(value=None):
    r = instance_resources(value)
    return ['--cpus', str(r['cpus']), '--memory', str(r['memory_gib'])+'g',
            '--memory-swap', str(r['memory_gib'])+'g', '--pids-limit', str(r['pids'])]


def inspect_resources(info, value=None):
    requested = instance_resources(value)
    expected = {'NanoCpus': requested['cpus']*10**9,
                'Memory': requested['memory_gib']*1024**3,
                'MemorySwap': requested['memory_gib']*1024**3,
                'PidsLimit': requested['pids']}
    hc = info.get('HostConfig', {})
    observed = {key: hc.get(key) for key in expected}
    mismatches = [key for key in expected
                  if type(observed[key]) is not int or observed[key] != expected[key]]
    return {'requested': requested, 'observed': observed,
            'compliant': not mismatches, 'mismatches': mismatches,
            'scope': 'daemon limits; CPU quota does not bound os.cpus() or prove runtime capacity'}
