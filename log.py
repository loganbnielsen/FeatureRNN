import logging

def setup_custom_logger(name, stdout_level):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    handler.setLevel(stdout_level)

    logger = logging.getLogger(name)
    logger.setLevel(logging.WARNING)
    logger.addHandler(handler)

    fh = logging.FileHandler('results.log', mode="w")
    fh.setFormatter(formatter)
    fh.setLevel(logging.WARNING)
    logger.addHandler(fh)

    return logger

setup_custom_logger('root', logging.WARNING)