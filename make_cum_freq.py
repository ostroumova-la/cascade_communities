import numpy as np
import sys

dst = open(sys.argv[1], "w")
results = np.loadtxt("freq.txt")
 
num = np.sum(results, axis = 0)[0]
total = num

sum = 0

for line in results:
    dst.write(str(line[1])+" "+str(total)+"\n")
    total -= line[0]
    sum += line[0]*line[1]

print(sum/num)
print((sum-results[0][0])/(num-results[0][0]))
print((num-results[0][0])/num)
    
dst.close()
