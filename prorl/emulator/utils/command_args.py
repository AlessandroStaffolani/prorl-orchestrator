from __future__ import print_function

from prorl.common.command_args import global_parser


def tim_dataset_commands(controller_parser):
    sub = controller_parser.add_parser('tim-dataset', help='TIM Dataset command')
    sub_subparsers = sub.add_subparsers(dest='action')
    create_parser = sub_subparsers.add_parser('create-index', help='Create the index data')
    create_parser.add_argument('output', help='Path where to save the indexes')
    global_parser(create_parser)
    upload_parser = sub_subparsers.add_parser('upload-index', help='Upload the index data to a MongoDB collection')
    upload_parser.add_argument('source_path', help='Indexes path')
    upload_parser.add_argument('collection_name', help='Collection name where the indexes will be uploaded')
    global_parser(upload_parser)


def add_emulator_parser(module_parser):
    sub = module_parser.add_parser('emulator', help='Emulator module')
    controller_parser = sub.add_subparsers(dest='controller')
    tim_dataset_commands(controller_parser)
