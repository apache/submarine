import logging 
from abc import ABCMeta 
from abc import abstractmethod 

logger = logging.getLogger(__name__) 

class AbstractModel(metaclass=ABCMeta): 
    @abstractmethod 
    def __init__(self): 
        pass 

    @abstractmethod
    def train(self): 
        pass 

    @abstractmethod
    def evaluate(self): 
        pass 

    @abstractmethod 
    def predict(self): 
        pass 


