# DATA_FOLDER = './data/common/ISR/datasets/'
# LOG_FOLDER = '/data/common/ISR/logs/'
DATA_FOLDER = './'
# LOG_FOLDER = '../logs/'
LOG_FOLDER = './logs'
from .model_config import model_attributes
from .train_config import get_train_command
from .parse_config import get_parse_command
