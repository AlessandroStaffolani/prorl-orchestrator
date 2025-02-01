from __future__ import print_function

import sys
from argparse import ArgumentParser

GLOBAL_ARGS = ['log_level', 'config_path']


def global_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument('--log-level', type=int, help='Log level. Default: 20 (INFO)')
    parser.add_argument('-cp', '--config-path', help='Path to config file')
    parser.add_argument('-mcp', '--multi-config-path', help='Path to multi run config file')
    return parser


def config_commands(main_parser):
    sub = main_parser.add_parser('config', help='Config command')
    sub_subparsers = sub.add_subparsers(dest='action')
    parser_sub_subparsers = sub_subparsers.add_parser('export', help='Export configuration')
    parser_sub_subparsers.add_argument('export_target', help='export target')
    parser_sub_subparsers.add_argument('-m', '--mode', default='yaml', help='Export format. Supported: yaml or json')
    parser_sub_subparsers.add_argument('--export-multi-run-config', action='store_true',
                                       help='Export the multi run config instead of the single run config')
    global_parser(parser_sub_subparsers)


def add_global_commands(main_parser):
    sub = main_parser.add_parser('global', help='Global module')
    controller_parser = sub.add_subparsers(dest='controller')
    config_commands(controller_parser)


def get_main_controller_args(args):
    module = args.module
    controller = args.controller
    action = args.action
    kwargs = get_action_arguments(args)
    return module, controller, action, kwargs


def get_action_arguments(args):
    kwargs = {}
    for key, value in vars(args).items():
        if key != 'module' and key != 'controller' and key != 'action' and key not in GLOBAL_ARGS:
            kwargs[key] = value
    return kwargs


def check_args(args, main_parser):
    if args.controller is None:
        print('Please select a command.', end='\n\n', file=sys.stderr)
        main_parser.print_help(file=sys.stderr)
        exit(2)
    elif args.action is None:
        print(f'Please select a {args.controller} action. Help: -h or --help', file=sys.stderr)
        exit(2)


def parse_arguments():
    main_parser = ArgumentParser(description='5G Load Balancer')
    main_parser = global_parser(main_parser)
    controller_parser = main_parser.add_subparsers(dest='controller')
    config_commands(controller_parser)
    args = main_parser.parse_args()
    check_args(args, main_parser)
    return args

