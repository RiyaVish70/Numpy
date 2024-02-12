```python
import numpy as np
x = np.array([1,2,3,4])
print(x)
print(type(x))
```

    [1 2 3 4]
    <class 'numpy.ndarray'>
    


```python
y = [1,2,3,4]    # this is list
print(y)
print(type(y))
```

    [1, 2, 3, 4]
    <class 'list'>
    

# difference between numpy array and list in python

# a.data type storage ,
# list me hum number srting both ko store karte hai but in numpy array only we store number or string , together not store
# for higher calcuation we store both no. and string

# b. importing module
# list ke liye koi bhi pakages module ko install nhi karna hota hai,we directly work on them.list is a inbuild type of datatype
# but in  numpy array we have to install pakages of numpy .to work on them 


# c. numerical operation
# numerical operation efficiency more in numpy than list

# d. modification capabilities
# numpy have less modificaion than list

# e. consumes less memory

# f. fast as compared to the python list

# g. convient to use


```python

```


```python
%timeit [j**4 for j in range(1,9)]
```

    1.65 µs ± 70 ns per loop (mean ± std. dev. of 7 runs, 1,000,000 loops each)
    


```python
import numpy as np
```


```python
%timeit np.arange(1,9)**4
```

    2.76 µs ± 66.1 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
    


```python
# in both cases we see which one faster 
```

# Array


```python
import numpy as np
x=[1,2,3,4]
y= np.array([1,2,3,4,5])
y

```




    array([1, 2, 3, 4, 5])




```python
l =[]
for i in range(1,5):
     int_1 = int(input("enter :"))
     l.append(int_1) #yha tk list create ho gi ,append kiye h list me

#ab array bnana chahte h so directly we create

print(np.array(l))
```

    enter :12
    enter :23
    enter :34
    enter :54
    [12 23 34 54]
    


```python
l =[]
for i in range(1,5):
     int_1 = input("enter :")
     l.append(int_1) #yha tk list create ho gi ,append kiye h list me

#ab array bnana chahte h so directly we create

print(np.array(l))
```

    enter :23
    enter :34
    enter :45
    enter :12
    ['23' '34' '45' '12']
    


```python
y= np.array([1,2,3,4,5])

```


```python
y.ndim
```




    1




```python
x = [[1,2,3]]
```


```python
x = np.array([[1,2,3]])
```


```python
x.ndim
```




    2




```python
x = np.array([[[[1,2,3],[1,2,3]]]])
print(x)
print(x.ndim)
```

    [[[[1 2 3]
       [1 2 3]]]]
    4
    


```python
x.ndim
```




    4




```python
x = np.array([[[[1,2,3,4],[1,2,3]]]])
print(x)
print(x.ndim)
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[36], line 1
    ----> 1 x = np.array([[[[1,2,3,4],[1,2,3]]]])
          2 print(x)
          3 print(x.ndim)
    

    ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 3 dimensions. The detected shape was (1, 1, 2) + inhomogeneous part.



```python
arn = np.array([1,2,3,4],ndmin=10)
print(arn)
print(arn.ndim)
```

    [[[[[[[[[[1 2 3 4]]]]]]]]]]
    10
    

# special numpy array
#              1.array fillled with 0's
#              2.array filled with 1's
#              3.create an empty array
#              4.an array with a range of element
#              5.array diagonal element filled with 1's
#              6.create an array with values that are spaced linearly in 
#                  a specified interval


```python
# o's array
```


```python
import numpy as np
ar_zero = np.zeros(4)
ar_zero1 = np.zeros((3,4))
print(ar_zero)
print()  # this for space only
print(ar_zero1)
```

    [0. 0. 0. 0.]
    
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
    


```python
# 1's array

```


```python
ar_one = np.ones(5)
print(ar_one)
```

    [1. 1. 1. 1. 1.]
    


```python
#empty array
```


```python
ar_empty = np.empty(4)
print(ar_empty)
```

    [0. 0. 0. 0.]
    


```python
# previously memory act like a empty array
```


```python
#range array
```


```python
ar_an = np.arange(4)
print(ar_an)    #arange is attribute 
```

    [0 1 2 3]
    


```python
# digonal filled with 1
```


```python
ar_dia = np.eye(3)  # eye attribute
print(ar_dia)
```

    [[1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]]
    


```python
ar_dia = np.eye(3,5)  # eye attribute
print(ar_dia)
```

    [[1. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0.]
     [0. 0. 1. 0. 0.]]
    


```python
# line space
```


```python
ar_lin = np.linspace(1,10,num=5) #(min,max,dimension)
print(ar_lin)
```

    [ 1.    3.25  5.5   7.75 10.  ]
    


```python
ar_lin = np.linspace(0,10,num=5)
print(ar_lin)
```

    [ 0.   2.5  5.   7.5 10. ]
    


```python
ar_lin = np.linspace(0,50,num=5)
print(ar_lin)
```

    [ 0.  12.5 25.  37.5 50. ]
    


```python
ar_lin = np.linspace(1,10,num=4)
print(ar_lin)
```

    [ 1.  4.  7. 10.]
    

# create numpy arrays with random numbers


```python
# rand()
```


```python
import numpy as np
var = np.random.rand(4)
print(var)
```

    [0.55633237 0.14014424 0.86356528 0.20321941]
    


```python
import numpy as np
var = np.random.rand(4)
print(var)
```

    [0.37593865 0.2255507  0.20059872 0.99446631]
    


```python
# it shows random value come 
```


```python
var1 = np.random.rand(2,5)

print(var1)
```

    [[0.98285369 0.48541605 0.23842915 0.34793379 0.77036618]
     [0.55766521 0.94312987 0.31914212 0.84389294 0.23876578]]
    


```python
#randn()
```


```python
var2 = np.random.randn(5)
```


```python
print(var2)
```

    [-1.40829175  0.7666666   1.627414    0.95625257 -1.24788111]
    


```python
#ranf()
```


```python
var3 = np.random.ranf(4)
```


```python
print(var3)
```

    [0.26235748 0.98965286 0.71615513 0.04100436]
    


```python
#randint()
```


```python
var4 = np.random.randint(5,20,5) #var4 = np.random.randint(min,max,total_values)

print(var4)
```

    [ 6 12  6 10 12]
    

# Data type


```python
import numpy as np
```


```python
var = np.array([1,2,3,4,12,15,30,32])
print("Data type:",var.dtype)
```

    Data type: int32
    


```python
var = np.array([1.0,2.0,3.0,4.0,1.2,1.5,3.0,3.2])
print("Data type:",var.dtype)
```

    Data type: float64
    


```python
var = np.array(["A","B","C","F"])
print("Data type:",var.dtype)
```

    Data type: <U1
    


```python
var = np.array(["A","B","C","F",1,2,3])
print("Data type:",var.dtype)  # U11 is a string data type
```

    Data type: <U11
    


```python
x = np.array([1,2,3,4],dtype=np.int8)
print("Data type:",x.dtype)
print(x)
```

    Data type: int8
    [1 2 3 4]
    


```python
x = np.array([1,2,3,4],)
print("Data type:",x.dtype) # kisi bhi data type to change kr sakte hai dtype = np.int8 ya int32 kuch bhi 
```

    Data type: int32
    


```python
x1 = np.array([1,2,3,4],dtype="f")
print("Data type:",x1.dtype)
print(x1)
```

    Data type: float32
    [1. 2. 3. 4.]
    


```python
x2 = np.array([1,2,3,4])
new = np.float32(x2)

print("Data Type : ",x2.dtype)
print("Data Type : ",new.dtype)
print(x2)
print(new)
```

    Data Type :  int32
    Data Type :  float32
    [1 2 3 4]
    [1. 2. 3. 4.]
    


```python
x2 = np.array([1,2,3,4])
new = np.float32(x2)

new_one = np.int_(new)

print("Data Type : ",x2.dtype)
print("Data Type : ",new.dtype)
print("Data Type : ",new_one.dtype)
print(x2)
print(new)
print(new_one)
```

    Data Type :  int32
    Data Type :  float32
    Data Type :  int32
    [1 2 3 4]
    [1. 2. 3. 4.]
    [1 2 3 4]
    


```python
x3 = np.array([1,2,3,4])
new_1 = x3.astype(float)
print(x3)
print(new_1)                                #with the help of astype function ,we change the type of any function
```

    [1 2 3 4]
    [1. 2. 3. 4.]
    

# shape and reshaping in numpy arrays

# shape


```python
import numpy as np
```


```python
var = np.array([[1,2],[1,2]])
print(var)
print()
print(var.shape)
```

    [[1 2]
     [1 2]]
    
    (2, 2)
    


```python
var1 = np.array([1,2,3,4],ndmin=4)
print(var1)
print()
print(var1.shape)
print(var1.ndim)
```

    [[[[1 2 3 4]]]]
    
    (1, 1, 1, 4)
    4
    

# reshape


```python
var2 = np.array([1,2,3,4,5,6])
print(var2)

print()

x = var2.reshape(3,2)
print(x)
print(x.ndim)
```

    [1 2 3 4 5 6]
    
    [[1 2]
     [3 4]
     [5 6]]
    2
    


```python
var2 = np.array([1,2,3,4,5,6])
print(var2)

print()

x = var2.reshape(3,3)
print(x)
```

    [1 2 3 4 5 6]
    
    


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[20], line 6
          2 print(var2)
          4 print()
    ----> 6 x = var2.reshape(3,3)
          7 print(x)
    

    ValueError: cannot reshape array of size 6 into shape (3,3)



```python
var2 = np.array([1,2,3,4,5,6,7,8,9])
print(var2)

print()

x = var2.reshape(3,3)
print(x)
print(x.shape)
print(x.ndim)
```

    [1 2 3 4 5 6 7 8 9]
    
    [[1 2 3]
     [4 5 6]
     [7 8 9]]
    (3, 3)
    2
    


```python
var3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
print(var3)
print(var3.ndim)
 
print()
    
x1 = var3.reshape(2,3,2)
print(x1)
print(x1.ndim)
print()
one = x1.reshape(-1)      # jaise hi -1 ko pass karwate hai wo one dimension array mechange ho jata
print(one)
print(one.ndim)
```

    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    1
    
    [[[ 1  2]
      [ 3  4]
      [ 5  6]]
    
     [[ 7  8]
      [ 9 10]
      [11 12]]]
    3
    
    [ 1  2  3  4  5  6  7  8  9 10 11 12]
    1
    

# Arithmetic Operation in numpy array


```python
import numpy as np
var = np.array([1,2,3,4])
var_1 = np.array([3,5,6,7])
varadd = var + var_1
print(varadd)
```

    [ 4  7  9 11]
    


```python
var = np.array([1,2,3,4])
varadd = var + 90
print(varadd)
```

    [91 92 93 94]
    


```python
var_1 = np.array([3,5,6,7])
varsub = var - 7
print(varsub)
```

    [-6 -5 -4 -3]
    


```python
var_2 = np.array([3,5,6,7])
varmul = var * 5
print(varmul)
```

    [ 5 10 15 20]
    


```python
var_3 = np.array([3,5,6,7])

vardiv = var % 2
print(vardiv)
```

    [1 0 1 0]
    


```python
var = np.array([1,2,3,4])
vardiv = var / 4
print(vardiv)
```

    [0.25 0.5  0.75 1.  ]
    

# use above function or

# 1. a+b   =   np.add(a,b)
# 2. a-b    =   np.subtract(a,b)
# 3. a*b    =   np.mutiply(a,b)
# 4. a/b    =    np.divide(a,b)
# 5. a%b  =   np.mod(a,b)
# 6. a**b   =   np.power(a,b)
# 7. 1/a     =   np.reciprocal(a)



```python

```


```python

```

# ARITHMETIC FUNCTIONS

# np.min(x)
# np.max(x)
# np.argmin(x)
# np.sqrt(x)
# np.sin(x)
# np.cos(x)
# np.cumsum(x)


```python
import numpy as np
var = np.array([1,2,3,4,5,6,7])
print("min:",np.min(var))
print("sqrt:",np.sqrt(var))

```

    min: 1
    sqrt: [1.         1.41421356 1.73205081 2.         2.23606798 2.44948974
     2.64575131]
    


```python
import numpy as np
var = np.array([1,2,3,4,10,8,12])
print("max:",np.max(var))
print("sqrt:",np.sqrt(var))
```

    max: 12
    sqrt: [1.         1.41421356 1.73205081 2.         3.16227766 2.82842712
     3.46410162]
    


```python
import numpy as np                   # axis = 0 (along column if need)
var = np.array([1,2,3,4,5,6,7])      # axis = 1 (along row if need)  
print("argmin:",np.argmin(var))
```

    argmin: 0
    


```python
var1 = np.array([[1,2,3],[7,8,9]])
print(np.min(var1,axis = 1))
```

    [1 7]
    


```python
var1 = np.array([[1,2,3],[7,8,9]])
print(np.min(var1,axis = 0))
```

    [1 2 3]
    


```python
import numpy as np
var = np.array([1,2,3,4,5,6,7])
print("sin:",np.sin(var))
print("cos:",np.cos(var))
print()
print(np.cumsum(var))  #diagonal element sum hota hai
```

    sin: [ 0.84147098  0.90929743  0.14112001 -0.7568025  -0.95892427 -0.2794155
      0.6569866 ]
    cos: [ 0.54030231 -0.41614684 -0.9899925  -0.65364362  0.28366219  0.96017029
      0.75390225]
    
    [ 1  3  6 10 15 21 28]
    


```python

```

# Broadcasting numpy arrays


```python
import numpy as np
var1 = np.array([1,2,3,4])
var2 = np.array([1,2,3])

print( var1 + var2 )
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[20], line 5
          2 var1 = np.array([1,2,3,4])
          3 var2 = np.array([1,2,3])
    ----> 5 print( var1 + var2 )
    

    ValueError: operands could not be broadcast together with shapes (4,) (3,) 



```python
var1 = np.array([1,2,3])
var2 = np.array([1,2,3])

print( var1 + var2 )
```

    [2 4 6]
    


```python
#operands could not be broadcast together with shapes (4,) (3,) 

# same dimension hona chahiye 
```


```python
var1 = np.array([1,2,3])
print(var1.shape)
print()
print(var1)
print()


var2 = np.array([[1],[2],[3]])
print(var2.shape)
print()
print(var2)

print()
print(var1 + var2)
```

    (3,)
    
    [1 2 3]
    
    (3, 1)
    
    [[1]
     [2]
     [3]]
    
    [[2 3 4]
     [3 4 5]
     [4 5 6]]
    


```python
x = np.array([[1],[2]])
print(x.shape)

print()

y = np.array([[1,2,3],[1,2,3]])
print(y.shape)

print(x+y)

```

    (2, 1)
    
    (2, 3)
    [[2 3 4]
     [3 4 5]]
    


```python

```

# indexing & slicing in numpy array


```python
import numpy as np
var = np.array([9,8,7,6])

#              0,1,2,3
#              -4,-3,-2,-1

print(var[1])
print(var[-3])
```

    8
    8
    


```python
var1 = np.array([[9,8,7],[3,4,5]])

print(var1)
print(var1.ndim)
print()

print(var1[0,1])
```

    [[9 8 7]
     [3 4 5]]
    2
    
    8
    


```python
var2 = np.array([[[1,2],[1,7]]])

print(var2)
print(var2.ndim)
print()

print(var2[0,1,1])
```

    [[[1 2]
      [1 7]]]
    3
    
    7
    

# slicing numpy arrays


```python
import numpy as np

var = np.array([1,2,3,4,5,6,7])
#              0,1,2,3,4,5,6
print(var)

print()

print("2 to 5 : ",var[1:5])
```

    [1 2 3 4 5 6 7]
    
    2 to 5 :  [2 3 4 5]
    


```python
import numpy as np

var = np.array([1,2,3,4,5,6,7])
#              0,1,2,3,4,5,6
print(var)

print()

print("2 to 5 : ",var[1:4]) #n-1 value give that is why we have to give one more indices extra for exact answer
```

    [1 2 3 4 5 6 7]
    
    2 to 5 :  [2 3 4]
    


```python
import numpy as np

var = np.array([1,2,3,4,5,6,7])
#              0,1,2,3,4,5,6
print(var)

print()

print("2 to 5 : ",var[1:5])

print("2 to end :", var[1:])
print("start to 5 :",var[:5])
print("stop : ",var[::2])
```

    [1 2 3 4 5 6 7]
    
    2 to 5 :  [2 3 4 5]
    2 to end : [2 3 4 5 6 7]
    start to 5 : [1 2 3 4 5]
    stop :  [1 3 5 7]
    


```python
import numpy as np

var = np.array([1,2,3,4,5,6,7])
#              0,1,2,3,4,5,6
print(var)

print()

print("2 to 5 : ",var[1:5])

print("2 to end :", var[1:])
print("start to 5 :",var[:5])
print("stop : ",var[1:6:2])
```

    [1 2 3 4 5 6 7]
    
    2 to 5 :  [2 3 4 5]
    2 to end : [2 3 4 5 6 7]
    start to 5 : [1 2 3 4 5]
    stop :  [2 4 6]
    

# two dimensional array


```python
var1 = np.array([[1,2,3,4],[2,3,4,5],[11,12,13,14]])
print(var1)
print()
print("8 to 5 :",var1[2,1:])
```

    [[ 1  2  3  4]
     [ 2  3  4  5]
     [11 12 13 14]]
    
    8 to 5 : [12 13 14]
    

# iterating numpy arrays


```python
import numpy as np
var = np.array([9,4,6,7,8,8])
print(var)
print()
for i in var :
    print(i)
```

    [9 4 6 7 8 8]
    
    9
    4
    6
    7
    8
    8
    


```python
var = np.array([[1,2,3,4],[4,7,9,0]])
print(var1)
print()
 
for j in var : 
    print(j)
    
print()

for k in var1:
    for l in k:
        print(l)
    
```

    [[ 1  2  3  4]
     [ 2  3  4  5]
     [11 12 13 14]]
    
    [1 2 3 4]
    [4 7 9 0]
    
    1
    2
    3
    4
    2
    3
    4
    5
    11
    12
    13
    14
    


```python
var3 = np.array([[[1,2,3],[4,5,6]]])
print(var3)
print()

for i in var3 :
    for k in i :
        for j in k : 
            print(j)
            
print()
print(var3.ndim)
```

    [[[1 2 3]
      [4 5 6]]]
    
    1
    2
    3
    4
    5
    6
    
    3
    


```python
# bar bar loop na chalane ka mn ho to ek bar me hi function ka use kr ke loop chla sakte hai
# nditer()
```


```python
var3 = np.array([[[1,2,3],[4,5,6]]])
print(var3)
print(var3.ndim)
print()

for i in np.nditer(var3,flags=['buffered'],op_dtypes=["S"]):
       print(i)

```

    [[[1 2 3]
      [4 5 6]]]
    3
    
    b'1'
    b'2'
    b'3'
    b'4'
    b'5'
    b'6'
    


```python
var3 = np.array([[[1,2,3],[4,5,6]]])
print(var3)
print()
for i in np.nditer(var3):
    print(i)
```

    [[[1 2 3]
      [4 5 6]]]
    
    1
    2
    3
    4
    5
    6
    


```python
var3 = np.array([[[3,4,5],[6,7,9]]])
print(var3)
print(var3.ndim)
print()

for i,d in np.ndenumerate(var3):
    print(i,d)
```

    [[[3 4 5]
      [6 7 9]]]
    3
    
    (0, 0, 0) 3
    (0, 0, 1) 4
    (0, 0, 2) 5
    (0, 1, 0) 6
    (0, 1, 1) 7
    (0, 1, 2) 9
    

# difference between copy and view in numpy array


```python
import numpy as np
var = np.array([1,2,3,4])

co = var.copy()

var[1]=40

print("var :",var)
print("copy :",co)
```

    var : [ 1 40  3  4]
    copy : [1 2 3 4]
    


```python
x = np.array([9,8,7,6,5])

vi = x.view()

x[1]=40

print("x :",x)
print("view :",vi)
```

    x : [ 9 40  7  6  5]
    view : [ 9 40  7  6  5]
    

# join and split function in numpy array


```python
import numpy as np

var  = np.array([3,4,5,6])
var1 = np.array([9,8,6,8])

ar = np.concatenate((var,var1)) # np.concatenate function use to join 
print(ar)
```

    [3 4 5 6 9 8 6 8]
    


```python
vr = np.array([[1,2],[3,4]])
vr1 = np.array([[8,9],[6,7]])

ar = np.concatenate((vr,vr1),axis=1)
print(ar)
```

    [[1 2 8 9]
     [3 4 6 7]]
    


```python
vr = np.array([[1,2],[3,4]])
vr1 = np.array([[8,9],[6,7]])

ar = np.concatenate((vr,vr1),axis=0)
print(ar)
```

    [[1 2]
     [3 4]
     [8 9]
     [6 7]]
    


```python
vr = np.array([[1,2,3,4]])
vr1 = np.array([[8,9,6,7]])

ar_new = np.stack((vr,vr1),axis = 1)
print(ar_new)
```

    [[[1 2 3 4]
      [8 9 6 7]]]
    


```python
vr = np.array([[1,2,3,4]])
vr1 = np.array([[8,9,6,7]])

ar_new = np.hstack((vr,vr1))  # h use to merge along row
print(ar_new)
```

    [[1 2 3 4 8 9 6 7]]
    


```python
vr = np.array([[1,2,3,4]])
vr1 = np.array([[8,9,6,7]])

ar_new = np.dstack((vr,vr1))  # d use to merge along height
print(ar_new)
```

    [[[1 8]
      [2 9]
      [3 6]
      [4 7]]]
    


```python
vr = np.array([[1,2,3,4]])
vr1 = np.array([[8,9,6,7]])

ar_new = np.vstack((vr,vr1))   # v use to merge along column 
print(ar_new)
```

    [[1 2 3 4]
     [8 9 6 7]]
    

# split array


```python
import numpy as np

var = np.array([1,2,3,4,5,6])

print(var)

ar = np.array_split(var,3)

print()
print(ar)
print(type(ar)) #check which type of ar is it

print(ar[0])
print()
print(ar[1])

```

    [1 2 3 4 5 6]
    
    [array([1, 2]), array([3, 4]), array([5, 6])]
    <class 'list'>
    [1 2]
    
    [3 4]
    

# search , sort , search sorted , filter   

# sort


```python
var_1 = np.array([4,2,3,1,12,5,22,52,6,7])
# index           0,1,2,3,4,5,6,7,8,9

print(np.sort(var_1))
```

    [ 1  2  3  4  5  6  7 12 22 52]
    


```python
var_2 = np.array(["a","z","m","f","d","k"])
# index           0,1,2,3,4,5,6,7
print(np.sort(var_2))
```

    ['a' 'd' 'f' 'k' 'm' 'z']
    


```python
var_1 = np.array([[4,2,3,1,12,5,22,52,6,7]])
# index           0,1,2,3,4,5,6,7,8,9

print(np.sort(var_1))
```

    [[ 1  2  3  4  5  6  7 12 22 52]]
    

# filter array


```python
var_3 = np.array(["a","z","m","f","d","k"])
# index           0,1,2,3,4,5

f= [True,False,True,True,False,True]

new_a = var_3[f]

print(new_a)

print(type(new_a))
```

    ['a' 'm' 'f' 'k']
    <class 'numpy.ndarray'>
    

# shuffle


```python
import numpy as np

var = np.array([1,2,3,4,5])

np.random.shuffle(var)
print(var)
```

    [3 5 1 2 4]
    

# unique


```python
var_1 = np.array([1,2,3,4,5,6,7,8,1,3,4,5,6])

x = np.unique(var_1,return_index=True,return_counts=True)

print(x)
```

    (array([1, 2, 3, 4, 5, 6, 7, 8]), array([0, 1, 2, 3, 4, 5, 6, 7], dtype=int64), array([2, 1, 2, 2, 2, 2, 1, 1], dtype=int64))
    

# resize


```python
var2 = np.array([1,2,3,4,5,6])

y = np.resize(var2,(2,3))

print(y)
```

    [[1 2 3]
     [4 5 6]]
    

# flatten


```python
var2 = np.array([1,2,3,4,5,6])

y = np.resize(var2,(2,3))

print(y)
print()

print("flatten :",y.flatten(order="F"))
print("revel : ",np.ravel(y,order="A"))
```

    [[1 2 3]
     [4 5 6]]
    
    flatten : [1 4 2 5 3 6]
    revel :  [1 2 3 4 5 6]
    

# Insert and delete function


```python
#insert
```


```python
import numpy as np

var = np.array([1,2,3,4])

print(var)

v = np.insert(var,2,40)
print(v)
```

    [1 2 3 4]
    [ 1  2 40  3  4]
    


```python
var = np.array([1,2,3,4])

print(var)

v = np.insert(var,(2,3),40)
print(v)
```

    [1 2 3 4]
    [ 1  2 40  3 40  4]
    


```python
var = np.array([1,2,3,4])

print(var)

v = np.insert(var,2,6.5) #it assign only inter not consider float value
print(v)
```

    [1 2 3 4]
    [1 2 6 3 4]
    


```python
var = np.array([1,2,3,4])

print(var)

#v = np.insert(var,2,40)

x = np.append(var,9.5)

print(x)
```

    [1 2 3 4]
    [1.  2.  3.  4.  9.5]
    


```python
var1 = np.array([[1,2,3],[44,6,7]])

v1 = np.insert(var1,2,[22,23,44], axis = 0)

print(v1)
```

    [[ 1  2  3]
     [44  6  7]
     [22 23 44]]
    

# Delete


```python
import numpy as np

var1 = np.array([1,2,3,4])

print(var1)

d = np.delete(var1,2)
print(d)
```

    [1 2 3 4]
    [1 2 4]
    

# MATRIX IN NUMPY ARRAYS


```python
import numpy as np

var = np.matrix([[1,2,3],[5,6,7]])

print(var)
print(type(var))
```

    [[1 2 3]
     [5 6 7]]
    <class 'numpy.matrix'>
    


```python
var1 = np.array([[1,2,3],[5,6,7]])

print(var1)
print(type(var1))
```

    [[1 2 3]
     [5 6 7]]
    <class 'numpy.ndarray'>
    

# function of matrix


```python
import numpy as np

var = np.matrix([[1,2,3],[4,6,7]])
print(var)
print()

print(np.transpose(var))
print()
print(var.T)#shortcut toperform transpose
```

    [[1 2 3]
     [4 6 7]]
    
    [[1 4]
     [2 6]
     [3 7]]
    
    [[1 4]
     [2 6]
     [3 7]]
    


```python
var = np.matrix([[1,2,3],[4,6,7]])
print(var)
print()

print(np.swapaxes(var,0,1))
```

    [[1 2 3]
     [4 6 7]]
    
    [[1 4]
     [2 6]
     [3 7]]
    


```python
var = np.matrix([[1,2],[6,7]])
print(var)
print()

print(np.linalg.inv(var))
```

    [[1 2]
     [6 7]]
    
    [[-1.4  0.4]
     [ 1.2 -0.2]]
    


```python
var = np.matrix([[1,2],[4,7]])
print(var)
print()

print(np.linalg.matrix_power(var,4))
print()
```

    [[1 2]
     [4 7]]
    
    [[ 593 1056]
     [2112 3761]]
    
    


```python
var = np.matrix([[1,2],[4,7]])
print(var)
print()

print(np.linalg.matrix_power(var,0))
print()
```

    [[1 2]
     [4 7]]
    
    [[1 0]
     [0 1]]
    
    


```python
var = np.matrix([[1,2,7],[4,7,8],[9,9,8]])
print(var)
print()

print(np.linalg.det(var))
print()
```

    [[1 2 7]
     [4 7 8]
     [9 9 8]]
    
    -124.99999999999994
    
    


```python
var = np.matrix([[1,2],[4,7]])
print(var)
print()

print(np.linalg.det(var))
print()
```

    [[1 2]
     [4 7]]
    
    -1.0
    
    
