from prorl.common.enum_utils import ExtendedEnum


class ModelTypes(str, ExtendedEnum):
    TimDatasetModel = 'tim-dataset-model'
    SyntheticModel = 'synthetic-model'
