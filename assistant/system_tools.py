"""Read-only local diagnostics shared by the executor and training catalog."""
import datetime
import json
import os
import platform
import shutil
from pathlib import Path


def disk_usage(params):
    path = Path(params.get('path', '~')).expanduser()
    usage = shutil.disk_usage(path)
    return True, json.dumps(dict(path=str(path), total_bytes=usage.total, used_bytes=usage.used, free_bytes=usage.free))


def system_info(params):
    return True, json.dumps(dict(system=platform.system(), release=platform.release(), machine=platform.machine(), hostname=platform.node(), cpu_count=os.cpu_count()))


def memory_status(params):
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key in {'MemTotal', 'MemAvailable', 'SwapTotal', 'SwapFree'}:
            values[key] = int(value.split()[0]) * 1024
    return True, json.dumps(values)


def uptime(params):
    seconds = int(float(Path('/proc/uptime').read_text().split()[0]))
    return True, str(datetime.timedelta(seconds=seconds))


HANDLERS = {'system.disk.usage': disk_usage, 'system.info': system_info,
            'system.memory.status': memory_status, 'system.uptime': uptime}
CATALOG = [
    ('system.disk.usage', {'path': 'string - filesystem path, defaults to home'}, 'Report disk capacity and free space', ['How much disk space is free?', 'Check storage usage', 'Show available disk space']),
    ('system.info', {}, 'Report OS, architecture, hostname and CPU count', ['What operating system is this?', 'Show system information', 'Tell me about this computer']),
    ('system.memory.status', {}, 'Report RAM and swap totals and availability in bytes', ['How much RAM is available?', 'Check memory usage', 'Show free memory']),
    ('system.uptime', {}, 'Report time since the computer booted', ['How long has this computer been running?', 'Show system uptime', 'Time since last boot?']),
]
