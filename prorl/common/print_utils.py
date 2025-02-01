

def print_status(current, total, pre_message='', loading_len=20, unit=''):
    perc = int((current * loading_len) / total)
    message = f'{pre_message}:\t{"#" * perc}{"." * (loading_len - perc)}\t{current}{unit}/{total}{unit}'
    print(message, end='\r')


def beautify_str(value: str, string_separator='_') -> str:
    split = value.split(string_separator)
    if len(split) == 0:
        split = value.split('-')
    return ' '.join(split).title()
