import logging
import os
import time
import sys
from datetime import datetime
from arg_parser import arguments

args = arguments()

t = time.time()
now = datetime.now()

mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)

report_dir = os.path.join(os.getcwd(), 'reports',
                          'report-'+now.strftime("%Y-%m-%d-%H-%M-%S")+str(args.name))
os.mkdir(report_dir)
os.mkdir(os.path.join(report_dir, "saved_model"))

logging.basicConfig(filename=os.path.join(
    report_dir, 'log.log'), level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


def log(text):
    print_string = "{:0.2f}s - {}"
    logging.info(print_string.format(time.time() - t, text))


def report_path(filename):
    return os.path.join(report_dir, filename)