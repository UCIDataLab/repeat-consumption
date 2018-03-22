"""
Wrapper for the Python logging package that also checks for level before logging.
The default version (which is uses here) starts with a INFO level and prints to the stdout.

Authors:
    1. Moshe Lichman
"""
import logging
import sys

_logger = logging.getLogger()
_logger.setLevel(logging.NOTSET)  # The logger should be set to the lowest level.

_ch = logging.StreamHandler(sys.stdout)
_ch.setLevel(logging.INFO)  # Default level.
_ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s --> %(message)s'))
_logger.addHandler(_ch)


def set_verbose():
    _ch.setLevel(logging.DEBUG)


def set_error():
    _ch.setLevel(logging.ERROR)


def debug(message):
    if _ch.level <= logging.DEBUG:
        _logger.debug(message)


def info(message):
    if _ch.level <= logging.INFO:
        _logger.info(message)


def warn(message):
    if _ch.level <= logging.WARNING:
        _logger.warning(message)


def error(message):
    if _ch.level <= logging.ERROR:
        _logger.error(message)


def critical(message):
    if _ch.level <= logging.CRITICAL:
        _logger.critical(message)
