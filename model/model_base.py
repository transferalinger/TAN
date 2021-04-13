"""
File: model/model_base.py
 - Contain base class for all other model classes.
"""
class ModelBase(object):
    """
    Description: base model for other model classes.
    """

    def __init__(self):
        """
        Clear all class variables.
        """
        self.log_format = ""
        self.train_f = None
        self.test_f = None
        self.best_f = None
        return

    def create_log_files(self, header):
        """
        Create log files.

        :param header: header of each files
        """
        self.train_f = open(self.log_dir + "/train.log", 'w')
        self.test_f = open(self.log_dir + "/test.log", 'w')
        self.best_f = open(self.log_dir + "/best.log", 'w')
        self.flush_logs()
        return

    def flush_logs(self):
        """
        Flush logs to log files.
        """
        self.train_f.flush()
        self.test_f.flush()
        self.best_f.flush()
        return

    def close_logs(self):
        """
        Close log files.
        """
        self.train_f.close()
        self.test_f.close()
        self.best_f.close()
        return

    def write_test(self, args):
        """
        Write args to test log file.

        :param args: text to write in test log file
        """
        self.test_f.write(self.log_format % args + '\n')
        
    def write_train(self, args):
        """
        Write args to train log file.

        :param args: text to write in train log file
        """
        self.train_f.write(self.log_format % args + '\n')

    def write_best(self, args):
        """
        Write args to best log file.

        :param args: text to write in best log file
        """
        self.best_f.write(self.log_format % args + '\n')
