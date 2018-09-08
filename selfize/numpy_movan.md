

```python
import numpy as np
```


```python
import pandas as pd
```


```python
array=np.array([[1,2,3],[2,3,4]])
```

numpy.array的三个方法：ndim,shape,size


```python
print('number of dimensiong; ',array.ndim)
```

    number of dimensiong;  2
    


```python
print('shape: ',array.shape)
```

    shape:  (2, 3)
    


```python
print('size:',array.size)
```

    size: 6
    

定义时指明矩阵元素类型，如int，int32，int64，float64等


```python
a=np.array([2,34,56],dtype=np.int64)
```


```python
print(a.dtype)
```

    int64
    


```python
a=np.array([[23,21,34],
           [67,34,12]])
print(a)
```

    [[23 21 34]
     [67 34 12]]
    

定义全0矩阵,全1矩阵


```python
a=np.zeros((3,4))
print(a)
```

    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    

empty方法，生成接近0的矩阵


```python
a=np.empty((3,4))
print(a)
```

    [[ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  0.  0.]]
    

生成有序序列


```python
a=np.arange(10,20,2)
print(a)
```

    [10 12 14 16 18]
    


```python
a=np.arange(12).reshape(3,4)
print(a)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
a=np.linspace(1,10,10).reshape((2,5))
print(a)
```

    [[  1.   2.   3.   4.   5.]
     [  6.   7.   8.   9.  10.]]
    

下面开始计算


```python
a=np.array([10,20,30,40])
b=np.arange(4)
print(a,b)
```

    [10 20 30 40] [0 1 2 3]
    


```python
c=a**b
print(c)
```

    [    1    20   900 64000]
    

函数运算


```python
c=10*np.sin(a)
print(c)
```

    [-5.44021111  9.12945251 -9.88031624  7.4511316 ]
    


```python
print(b<3)
```

    [ True  True  True False]
    

矩阵的两种乘法


```python
m=np.array([[1,2],
           [3,4]])
n=np.arange(4).reshape((2,2))
print(m)
print(n)
```

    [[1 2]
     [3 4]]
    [[0 1]
     [2 3]]
    

星乘是点乘，点乘是叉乘


```python
p=m*n
q=np.dot(m,n)
print(p)
print(q)
```

    [[ 0  2]
     [ 6 12]]
    [[ 4  7]
     [ 8 15]]
    


```python
q_2=m.dot(n)
print(q_2)
```

    [[ 4  7]
     [ 8 15]]
    

随机矩阵


```python
a=np.random.random((2,4))
print(a)
```

    [[ 0.00972206  0.45065498  0.88553621  0.76224452]
     [ 0.27213714  0.33010275  0.70174575  0.37757628]]
    


```python
np.sum(a)
```




    3.7897197005915668




```python
np.min(a)
```




    0.0097220570885678814




```python
np.max(a)
```




    0.8855362121908793



行列运算


```python
a=np.array([[1,2,3],
           [2,3,4]])
```

行列运算,axis=1对行运算，0对列运算


```python
print(np.sum(a,axis=1))#行列运算,axis=1对行运算，0对列运算
```

    [6 9]
    


```python
print(np.min(a,axis=0))
```

    [1 2 3]
    


```python
A=np.arange(2,14).reshape(3,4)
print(A)
```

    [[ 2  3  4  5]
     [ 6  7  8  9]
     [10 11 12 13]]
    


```python
print(np.argmax(A))
```

    11
    


```python
print(np.mean(A,axis=1))
```

    [  3.5   7.5  11.5]
    


```python
print(np.average(A))
```

    7.5
    


```python
print(np.median(A))#中位数
```

    7.5
    


```python
print(np.cumsum(A)) #逐个累加
```

    [ 2  5  9 14 20 27 35 44 54 65 77 90]
    


```python
print(np.diff(A))#亮亮之差
```

    [[1 1 1]
     [1 1 1]
     [1 1 1]]
    


```python
print(np.nonzero(A))
```

    (array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))
    


```python
A=np.arange(14,2,-1).reshape(3,4)
print(A)
```

    [[14 13 12 11]
     [10  9  8  7]
     [ 6  5  4  3]]
    


```python
print(np.sort(A))#排序
```

    [[11 12 13 14]
     [ 7  8  9 10]
     [ 3  4  5  6]]
    


```python
print(np.transpose(A))#转置
```

    [[14 10  6]
     [13  9  5]
     [12  8  4]
     [11  7  3]]
    


```python
print((A.T))#转置，简便写法
```

    [[14 10  6]
     [13  9  5]
     [12  8  4]
     [11  7  3]]
    


```python
print(np.clip(A,5,9))#设定上下限
```

    [[9 9 9 9]
     [9 9 8 7]
     [6 5 5 5]]
    


```python
A=np.arange(3,15).reshape((3,4))
print(A)
```

    [[ 3  4  5  6]
     [ 7  8  9 10]
     [11 12 13 14]]
    


```python
print(A[2])
```

    [11 12 13 14]
    


```python
print(A[2][1])
```

    12
    


```python
print(A[2,1])
```

    12
    


```python
print(A[:,1])
```

    [ 4  8 12]
    


```python
print(A[1,1:3])
```

    [8 9]
    


```python
for row in A:
    print(row)
```

    [3 4 5 6]
    [ 7  8  9 10]
    [11 12 13 14]
    


```python
for column in A.T:#numpy没有直接按列输出的方法，可用转置来搞定
    print(column.T)
```

    [ 3  7 11]
    [ 4  8 12]
    [ 5  9 13]
    [ 6 10 14]
    


```python
print(A.flatten())
```

    [ 3  4  5  6  7  8  9 10 11 12 13 14]
    


```python
for item in A.flat:
    print(item)
```

    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    


```python
A=np.array([1,1,1])#矩阵的合并
B=np.array([2,2,2])
print(np.vstack((A,B)))
```

    [[1 1 1]
     [2 2 2]]
    


```python
C=np.vstack((A,B))
```


```python
print(A.shape,C.shape)
```

    (3,) (2, 3)
    


```python
print(np.hstack((A,B)))
```

    [1 1 1 2 2 2]
    


```python
A.T#行向量转置无法实现列向量
```




    array([1, 1, 1])




```python
print(A[:,np.newaxis])#用newaxis实现列向量
```

    [[1]
     [1]
     [1]]
    


```python
A=np.array([1,1,1])[:,np.newaxis]
B=np.array([2,2,2])[:,np.newaxis]
print(np.hstack((A,B,B,A)))
```

    [[1 2 2 1]
     [1 2 2 1]
     [1 2 2 1]]
    


```python
E=np.concatenate((A,B,B,A),axis=1)#对四个向量拼接，axis=0为纵向，=1为横向
print(E)
```

    [[1 2 2 1]
     [1 2 2 1]
     [1 2 2 1]]
    


```python
E=np.hstack((A,B,B,A))
print(E)
```

    [[1 2 2 1]
     [1 2 2 1]
     [1 2 2 1]]
    


```python
F=np.concatenate((E),axis=0)
print(F)                                            
```

    [1 2 2 1 1 2 2 1 1 2 2 1]
    


```python
A=np.arange(12).reshape((3,4))#下面开始矩阵分割
print(A)
```

    [[ 0  1  2  3]
     [ 4  5  6  7]
     [ 8  9 10 11]]
    


```python
C=np.split(A,3,axis=0)
print(C)
```

    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    


```python
D=np.array_split(A,3,axis=1)#不等量分割
print(D)
```

    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2],
           [ 6],
           [10]]), array([[ 3],
           [ 7],
           [11]])]
    


```python
E=np.vsplit(A,3)
F=np.hsplit(A,2)
print(E)
print(F)
```

    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]]), array([[ 8,  9, 10, 11]])]
    [array([[0, 1],
           [4, 5],
           [8, 9]]), array([[ 2,  3],
           [ 6,  7],
           [10, 11]])]
    


```python
a=np.arange(4)#矩阵的赋值
b=a           #"="表示完全复制
print(a)
```

    [0 1 2 3]
    


```python
a[0]=11
```


```python
a
```




    array([11,  1,  2,  3])




```python
b
```




    array([11,  1,  2,  3])




```python
b is a
```




    True




```python
b=a.copy()    #deep copy，只复制值
```


```python
a[3]=12
print(a)
print(b)
```

    [11  1  2 12]
    [11  1  2  3]
    
