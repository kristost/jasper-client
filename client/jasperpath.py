# -*- coding: utf-8-*-
import os
from datetime import datetime

# Jasper main directory
APP_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir))

DATA_PATH = os.path.join(APP_PATH, "static")
LIB_PATH = os.path.join(APP_PATH, "client")
PLUGIN_PATH = os.path.join(LIB_PATH, "modules")

CONFIG_PATH = os.path.expanduser(os.getenv('JASPER_CONFIG', '~/.jasper'))


def config(*fname):
    return os.path.join(CONFIG_PATH, *fname)


def data(*fname):
    return os.path.join(DATA_PATH, *fname)

# This really shouldn't be here, but putting it here
# will allow access without having to touch every file that needs it.
# Kludgy, I know.
def get_timestamp(fmt='%d%m%YT%H%M%S'):
    return datetime.now().strftime(fmt)
