from prody import LOGGER
import logging.config


__all__ = ['sampling.generate_peptides.PeptideSampler']


LOGGER.verbosity = 'ERROR'

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'default': {
            'format': '[%(levelname)s] %(asctime)s [pid %(process)d] - %(message)s'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
        }
    },
    'loggers': {
        'console': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': False,
        }
    }
}

logging.config.dictConfig(LOGGING)
