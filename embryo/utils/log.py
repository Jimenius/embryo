'''Log utilities

Created by Minhui Li on June 11, 2020
'''


import functools
import logging
import os
import sys


def setup(
    level: int = logging.DEBUG,
    directory: str = '',
) -> None:
    '''Setup root logging settings

    Will create the directory if specified.
    Logs will be presented on stdout and in text file log.txt created in the directory.

    Args:
    directory: str
        Log directory
    '''

    # Logging formatter
    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
    handlers = []

    # StreamHandler puts message to stdout
    sh = logging.StreamHandler(stream = sys.stdout)
    sh.setFormatter(formatter)
    handlers.append(sh)

    # FileHandler puts message to file
    if directory:
        os.makedirs(directory, exist_ok=True)
        fh = logging.FileHandler(os.path.join(directory, 'log.txt'))
        fh.setFormatter(formatter)
        handlers.append(fh)

    logging.basicConfig(level=level, handlers=handlers)
