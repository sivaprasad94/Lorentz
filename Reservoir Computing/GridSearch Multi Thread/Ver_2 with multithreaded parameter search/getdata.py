import numpy as np
import matplotlib.pyplot as plt
#from pyESN import ESN
import ESN
# import ESNold as ESN
# import reservoir as ESN

np1=np.zeros([8329,25])
for i in range(1,26):
    list1=[]
    f=open("data/"+str(i)+".txt","r")
    for j in range(5):
        f.readline()
    for j in range(8329):
        data=f.readline().split()
        #print(data[1])
        np1[j][i-1]=data[1]

t=open("2.txt","w")

for i in range(8329):
    st=""
    for j in range(25):
        st=st+str(np1[i][j])+" "
    st=st+"\n"
    t.write(st)

#print(np1)








f.close()