'''
C 2
D 3
F 5
s 18
v 21
x 23
'''

lettere_nostre = [2,3,5,18,21,23]
newLables = {
    '2': '0',
    '3': '1',
    '5': '2',
    '18': '3',
    '21': '4',
    '23': '5'
    }


def AZTrimmer():
    file1 = open('A_Z Handwritten Data.csv', 'r') 
    
    while True: 
        line = file1.readline() 
    
        # if line is empty 
        # end of file is reached 
        if not line: 
            break
        
        label = line.split(',')[0]
        
        if int(label) in lettere_nostre: 
            line = line.replace(label, newLables[label], 1)
            with open("trimmedData.csv", "a") as fp2: 
                fp2.writelines(line) 
    
    file1.close() 


def main():
    AZTrimmer()


if __name__ == "__main__":
    main()
