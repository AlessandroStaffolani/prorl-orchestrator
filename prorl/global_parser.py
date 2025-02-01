from argparse import ArgumentParser

from prorl.common.command_args import global_parser, check_args, add_global_commands
from prorl.emulator.utils.command_args import add_emulator_parser
from prorl.run.utils.command_args import add_run_parser


def global_parse_arguments():
    main_parser = ArgumentParser(description='PRORL Orchestrator')
    global_parser(main_parser)
    module_parser = main_parser.add_subparsers(dest='module', title='Module', required=True)
    add_emulator_parser(module_parser)
    add_run_parser(module_parser)
    add_global_commands(module_parser)
    args = main_parser.parse_args()
    check_args(args, main_parser)
    return args
