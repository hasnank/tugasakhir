k=0
for line in reversed(list(open("filename"))):
     #print(line.rstrip())
     i=0
     for letter in line: 
         if letter == '9':
            print(k,",",i,sep="") 
         i=i+1
     k=k+1    
