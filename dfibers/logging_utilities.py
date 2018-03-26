import os

class Logger:

    def __init__(self, logfile, prefix=""):
        """
        Initialize logger with prefix and a file open for writing
        """
        self.logfile = logfile
        self.prefix = prefix

    def log(self, data):
        """
        Write data to log file
        """
        if self.logfile.name == os.devnull: return
        self.logfile.write(self.prefix + data)
        self.logfile.flush()
        # os.fsync(self.logfile)

    def plus_prefix(self, prefix):
        """
        Construct new Logger with same prefix as self plus new one
        """
        return Logger(self.logfile, self.prefix + prefix)
