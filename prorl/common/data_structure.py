from prorl.common.enum_utils import ExtendedEnum


class RunMode(str, ExtendedEnum):
    Train = 'training'
    Validation = 'validation'
    Eval = 'evaluation'
