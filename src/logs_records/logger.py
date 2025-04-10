import logging
import os
from datetime import datetime

LOG_FILE=f"{datetime.now()}.txt"

logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE)
os.makedirs(logs_path,exist_ok=True) #if exists dont create again

LOG_FILE_PATH=os.path.join(logs_path,LOG_FILE)

logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s -%(levelname)s - %(message)s",
    level=logging.DEBUG,
    filemode='w'


)


