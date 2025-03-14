import numpy as np
import matplotlib.pyplot as plt



data = np.loadtxt('data.csv', delimiter=',', skiprows= 1)

print (f'Number of persons tested {data.shape[0]}')

plt.scatter(data[:,1], data[:,2], label= 'Everyone tested', color='g')
plt.xlabel('Height(cm)')
plt.ylabel('Weight(kg)')
plt.show()



if data.shape[0] >= 50:
    plt.scatter(data[::50,1], data[::50,2],label='For every 50th person',color='b')
    plt.xlabel('Height(cm)')
    plt.ylabel('Weight(kg)')
    plt.show()

print(f'Minimal height is {data[:,1].min()}')
print(f'Maximal height is {data[:,1].max()}')
print(f'Mean height is {data[:,1].mean()}')


man = data[data[:,0]==1] 
woman = data[data[:,0] == 0] 

print(f'Tallest man is {man[:,1].max()} cm')
print(f'shortest man is {man[:,1].min()} cm')
print(f'Average man is {man[:,1].mean()} cm')

print(f'Tallest woman is {woman[:,1].max()} cm')
print(f'Shortest woman is {woman[:,1].min()} cm')
print(f'Average woman is {woman[:,1].mean()} cm')