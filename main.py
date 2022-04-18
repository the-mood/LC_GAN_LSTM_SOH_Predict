import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
list=[]
for i in range(16):
    list.append(i)

tranfer=StandardScaler()
list_tran=tranfer.fit_transform(np.array(list).reshape(-1,1))
list_inver_tran=tranfer.inverse_transform(np.array(list_tran).reshape(-1,1))
print(list_tran)
print('-'*30)
print(list_inver_tran)
print('*'*30)
t=MinMaxScaler()
list_M_t=t.fit_transform(np.array(list).reshape(-1,1))
list_inver_M=t.inverse_transform(np.array(list_M_t).reshape(-1,1))
print(list_M_t)
print('-'*30)
print(list_inver_M)