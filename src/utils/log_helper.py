import os
import sys
import logging
import errno
import datetime
from pytz import timezone, utc


class logHelper():
    """
    Helper class to configure logger
    """
    log_format = '%(asctime)s %(levelname)s - %(name)s - %(message)s'

    @staticmethod
    def setup(path, log_level='INFO'):
        logging.basicConfig(
            filename = path, 
            level = logging.getLevelName(log_level), 
            format = logHelper.log_format)

        def customTime(*args):
            utc_dt = utc.localize(datetime.datetime.utcnow())
            my_tz = timezone('Asia/Hong_Kong')
            converted = utc_dt.astimezone(my_tz)
            return converted.timetuple()
        
        logging.Formatter.converter = customTime

        # Set up logging to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(logging.Formatter(logHelper.log_format))

        # Add console handler to root logger
        logging.getLogger('').addHandler(console)

        # Log for unhandled exception
        logger = logging.getLogger(__name__)
        sys.excepthook = lambda *ex: logger.critical('Unhandled exception', exc_info=ex)

        logger.info('Finished configuring logger')

