'''
C 2
D 3
F 5
s 18
v 21
x 23
'''

lettere_spastiche = [2,3,5,18,21,23]

file1 = open('A_Z Handwritten Data.csv', 'r') 
  
while True: 
    line = file1.readline() 
  
    # if line is empty 
    # end of file is reached 
    if not line: 
        break
     
    label = line.split(',')[0]
    
    if int(label) in lettere_spastiche: 
        with open("trimmedData.csv", "a") as fp2: 
            fp2.writelines(line) 
  
file1.close() 