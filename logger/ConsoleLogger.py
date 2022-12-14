import logging

class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    green = "\033[92m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    # format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

class MyColorConsoleLogger():
    def __init__(self, process_name="Training Process", log_level="debug"):
        # create logger with 'spam_application'
        self.logger = logging.getLogger(process_name)
        if log_level=="debug":
            self.logger.setLevel(logging.DEBUG)
        elif log_level=="info":
            self.logger.setLevel(logging.INFO)

        # create console handler with a higher log level
        ch = logging.StreamHandler()
        if log_level=="debug":
            ch.setLevel(logging.DEBUG)
        elif log_level=="info":
            ch.setLevel(logging.INFO)

        ch.setFormatter(CustomFormatter())
        self.logger.addHandler(ch)

    def debug(self, msg):
        self.logger.debug(msg)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)
    
    def error(self, msg):
        self.logger.error(msg)
    
     
    def critical(self, msg):
        self.logger.critical(msg)