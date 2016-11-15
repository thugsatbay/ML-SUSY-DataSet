import numpy as np
'''
x=np.array([1,2,3,4,5,6,7,8]).reshape(8,1)
x=np.insert(x,0,0).reshape(9,1)
print x.shape
print x
a=np.array([[1,2,3],[4,5,6],[7,8,9]])
a=np.insert(a,0,1,axis=1)
print a
'''
q=np.array([[1,2,3],[4,5,6]])
print np.sum(q,axis=0)