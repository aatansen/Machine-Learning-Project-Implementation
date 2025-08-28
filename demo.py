from us_visa_approval_prediction.logger import logging
from us_visa_approval_prediction.exception import USvisaException
import sys
logging.info("Starting the demo script...")

try:
    r = 3/0
    print(r)
except Exception as e:
    logging.info(e)
    raise USvisaException(e, sys)