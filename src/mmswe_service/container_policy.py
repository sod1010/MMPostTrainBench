"""Operator-selected profiles with identical required isolation, never a retry.

The platform-managed profile requires equivalent daemon fields. The explicit
entry-guard profile defers NNP/capability checks to a pinned static entrypoint
and a host-approved handshake before candidate execution. Other constraints
still apply before container start. No profile is selected as a fallback.
"""

PROFILES = ('strict-v1', 'platform-managed-v1', 'entry-guard-v1')


def profile_name(value='strict-v1'):
    if value not in PROFILES:
        raise ValueError('unknown MMSWE container profile')
    return value


def security_args(kind, profile='strict-v1'):
    profile_name(profile)
    if kind not in ('workload', 'instance'):
        raise ValueError('unknown container kind')
    if profile in ('platform-managed-v1', 'entry-guard-v1'):
        return []
    drops = ['ALL'] if kind == 'workload' else ['NET_RAW', 'MKNOD']
    result = ['--ipc', 'private', '--security-opt', 'no-new-privileges=true']
    for capability in drops:
        result += ['--cap-drop', capability]
    return result


def inspect_security(info, kind, profile='strict-v1'):
    """Return a bounded, credential-free receipt, including rejection reasons."""
    security_args(kind, profile)  # validate even in managed mode
    hc, config = info.get('HostConfig', {}), info.get('Config', {})
    options = hc.get('SecurityOpt') or []
    drops = hc.get('CapDrop') or []
    guarded = profile == 'entry-guard-v1'
    problems = []
    if hc.get('Privileged') is not False:
        problems.append('privileged must be false')
    if hc.get('CapAdd'):
        problems.append('additional capabilities are forbidden')
    if hc.get('NetworkMode') != 'none':
        problems.append('network must be none')
    if hc.get('IpcMode') != 'private':
        problems.append('IPC must be private')
    if hc.get('PidMode') not in ('', 'private') or hc.get('UTSMode') not in ('', 'private'):
        problems.append('host/shared PID or UTS namespace is forbidden')
    if hc.get('GroupAdd'):
        problems.append('additional groups are forbidden')
    normalized = {str(x).replace(':', '=', 1) for x in options}
    if not guarded and not normalized.intersection({'no-new-privileges', 'no-new-privileges=true'}):
        problems.append('no-new-privileges is not enforced in inspect')
    if any(x.endswith('=unconfined') or x == 'no-new-privileges=false' for x in normalized):
        problems.append('disabled security policy is forbidden')
    dropped = {str(x).upper().removeprefix('CAP_') for x in drops}
    required = {'ALL'} if kind == 'workload' else {'NET_RAW', 'MKNOD'}
    if not guarded and 'ALL' not in dropped and not required.issubset(dropped):
        problems.append('required capability drops are not enforced in inspect')
    expected_user = '65534:65534' if kind == 'workload' and not guarded else '0:0'
    if config.get('User') != expected_user:
        problems.append('unexpected container user')
    if guarded:
        from .runtime_guard import GUARD_PATH, command
        args = config.get('Cmd')
        try:
            if (config.get('Entrypoint') != [GUARD_PATH] or not isinstance(args, list) or len(args) != 3
                    or args[0] != kind or args != command(*args)
                    or config.get('OpenStdin') is not True or config.get('Tty') is not False):
                raise ValueError('invalid guard')
        except (ValueError, TypeError):
            problems.append('fixed interactive entry guard is required')
    if kind == 'workload' and hc.get('ReadonlyRootfs') is not True:
        problems.append('workload root filesystem must be read-only')
    if kind == 'instance' and (info.get('Mounts') or hc.get('Binds') or hc.get('Mounts')):
        problems.append('instance containers must not mount host or volume data')
    observed = {key: hc.get(key) for key in (
        'Privileged', 'CapAdd', 'CapDrop', 'SecurityOpt', 'NetworkMode', 'IpcMode',
        'PidMode', 'UTSMode', 'UsernsMode', 'GroupAdd', 'ReadonlyRootfs')}
    observed.update(user=config.get('User'), apparmor=info.get('AppArmorProfile'),
                    image=info.get('Image'), mount_count=len(info.get('Mounts') or []))
    return {'version': 1, 'profile': profile, 'kind': kind,
            'compliant': not problems, 'violations': problems, 'observed': observed,
            'runtime_guard_required': guarded,
            'scope': 'daemon configuration; real runtime acceptance still required'}
