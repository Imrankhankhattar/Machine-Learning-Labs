import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data=datasets.load_boston()
df=pd.DataFrame(data["data"],columns=data["feature_names"])
print(df.describe)