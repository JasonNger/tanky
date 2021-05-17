import numpy as np
label1 = np.load('a.npy')
f = open('test1.txt')
end = open('end.txt', 'w')
j = 0
for i in f:
    end.write(i.split('\n')[0]+' '+str(int(label1[j]==0))+'\n')
    j+=1

f.close()
end.close()