from math import sqrt
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import random 

x_start=100
y_start=100
x_prev=x_start
y_prev=y_start
global x_current,y_curent

vel_x=0.5
vel_y=0.5
frames=10**7
distance_register=[]
for i in range(0,frames,1):
    x_current=x_prev+random.choice([1, -1])*vel_x
    y_curent=y_prev+random.choice([1, -1])*vel_y
    distance_register.append(sqrt((x_current-x_start)**2+(y_curent-y_start)**2))
    x_prev=x_current
    y_prev=y_curent
    if ((x_current-x_start)**2+(y_curent-y_start)**2>1000):
        break
plt.plot(distance_register)
plt.show()
