import datetime
import logging

def mylogger(name: str, rank=None):
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

    if rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger