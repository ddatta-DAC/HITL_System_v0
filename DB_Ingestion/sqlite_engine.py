from sqlalchemy import create_engine
import pandas as pd

class sqlite:

    engine = create_engine('sqlite:///./../DB/wwf.db', echo=False)
    def __init__(self):
        return 
    
    @staticmethod
    def get_engine():
        return sqlite.engine
