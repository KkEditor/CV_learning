import numpy as np

def DFT(x):
    N=len(x)
    res=np.zeros(N,dtype='complex')
    for r in range(N):
        for k in range(N):
            res[r] += x[k] * np.exp(-2j * np.pi * k * r / N)
    return res
def iDFT(x):
    N=len(x)
    res=np.zeros(N,dtype='complex')
    for k in range(N):
        for r in range(N):
            res[k] += 1/N * x[r] * np.exp (2j *np.pi * k * r /N)
    return res
x = np.asarray([2, 3, -1, 1], dtype='float')
print(DFT(x))
print(iDFT(DFT(x)))