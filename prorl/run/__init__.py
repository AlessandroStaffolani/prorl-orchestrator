from prorl.common.enum_utils import ExtendedEnum


class RunStatus(str, ExtendedEnum):
    WAITING = 'WAITING'
    SCHEDULED = 'SCHEDULED'
    RUNNING = 'RUNNING'
    EXECUTED = 'EXECUTED'
    COMPLETED = 'COMPLETED'
    ERROR = 'ERROR'
