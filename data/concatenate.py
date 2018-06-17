# -*- coding:utf8 -*-
import pandas as pd

SaveFile_Name = 'data_all.csv'

df = pd.read_csv('train.csv', header=None, sep='\t')
df.to_csv(SaveFile_Name,encoding="utf_8_sig", index=False, header=False, sep='\t')
df = pd.read_csv('train_add.csv', header=None, sep='\t')
df.to_csv(SaveFile_Name, encoding="utf_8_sig", index=False, header=False, sep='\t', mode='a+')
