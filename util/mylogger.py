import datetime
import logging

def mylogger(name: str, rank=0, is_initialized=False):
    logger = logging.getLogger(name)
    logger.setLevel("INFO")
    logger.propagate = False

    logfile = "trainingLog/"+name+".txt"

    def _utc8_aera(timestamp):
        now = datetime.datetime.utcfromtimestamp(timestamp) + datetime.timedelta(hours=8)
        return now.timetuple()

    formatter = logging.Formatter('[%(asctime)s]-[%(name)s:%(levelname)s]:%(message)s')
    formatter.converter = _utc8_aera

    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # if the rank is 0, then also print the training process on the console
    # if is not in multi GPU training, print on the console
    if not is_initialized or rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger