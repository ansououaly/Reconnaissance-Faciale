import pandas as pd
from datetime import datetime
import os

def log_recognition(name):
    os.makedirs("data", exist_ok=True)
    df = pd.DataFrame([[name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]],
                      columns=["Nom", "Date"])
    path = "data/journal.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(path, index=False)