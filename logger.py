import logging
from rich.logging import RichHandler
import time

logger = logging.getLogger(__name__)


# the handler determines where the logs go: stdout/file
#shell_handler = logging.StreamHandler()

timestr = time.strftime("%Y_%m_%d_%H_%M_%S")

shell_handler = RichHandler()
file_handler = logging.FileHandler(f"{timestr}.log")


logger.setLevel(logging.INFO)
shell_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.INFO)

# the formatter determines what our logs will look like
# fmt_shell = '%(levelname)s %(asctime)s %(message)s'
fmt_shell = '%(message)s'
fmt_file = '%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s'

shell_formatter = logging.Formatter(fmt_shell)
file_formatter = logging.Formatter(fmt_file)

# here we hook everything together
shell_handler.setFormatter(shell_formatter)
file_handler.setFormatter(file_formatter)

logger.addHandler(shell_handler)
logger.addHandler(file_handler)