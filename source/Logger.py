import logging

class Logger:
    logger = logging.getLogger("process_logger")
    logger.setLevel(logging.INFO)
    def log_message(self, message):
        self.logger.info(message)