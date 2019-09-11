import os
import logging.config

config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logging.conf')
logging.config.fileConfig(config_path)
