import random
import math

def sigmoid(x):
    y = 1 / (1 + math.exp(-x))
    return y

def sigmoid_matrix(matrix):
    print(matrix)
    ret = []
    for i in range(len(matrix)):
        temp = []
        for j in range(len(matrix[i])):
            temp.append(sigmoid(matrix[i][j]))
        ret.append(temp)
    return ret

def sign(number):
    if(number<0):
        return -1
    else:
        return

# dot product of two matrix
def hadamardPro(mat1, mat2):
    if(len(mat1)==len(mat2)):
        ret = []
        for i in range(len(mat1)):
            ret.append(mat1[i]*mat2[i])
        return ret
    else:
        print("Dimensions of matrix for hadamard do not match!")

def determinant(matrix):
    if(len(matrix)==2):
        return matrix[0][0]*matrix[1][1] - matrix[0][1]*matrix[1][0]
    else:
        summ = 0
        for i in range(len(matrix[0])):
            tempmat = []
            for j in range(len(matrix)-1):
                j+=1
                temp=[]
                for k in range(len(matrix[j])):
                    if(k!=i):
                        temp.append(matrix[j][k])
                tempmat.append(temp)
            summ += matrix[0][i]*math.pow(-1, i)*determinant(tempmat)
        return summ

def adjointMatrix(matrix):
    retmat = []
    for l in range(len(matrix)):
        againtemp = []
        for m in range(len(matrix[l])):
            tempmat = []
            for j in range(len(matrix)):
                if(j!=l):
                    temp=[]
                    for k in range(len(matrix[j])):
                        if(k!=m):
                            temp.append(matrix[j][k])
                    tempmat.append(temp)
            againtemp.append(math.pow(-1, (l+m))*determinant(tempmat))
        retmat.append(againtemp)
    return retmat

def transpose(matrix):
    ret = []
    for i in range(len(matrix[0])):
        temp = []
        for j in range(len(matrix)):
            temp.append(matrix[j][i])
        ret.append(temp)
    return ret

def inverse(matrix):
    newmat = adjointMatrix(matrix)
    newmat = transpose(newmat)
    det = determinant(matrix)
    if(det!=0):
        for i in range(len(newmat)):
            for j in range(len(matrix[i])):
                newmat[i][j] = newmat[i][j]/det
        return newmat
    else:
        print('The matrix is non invertible!!')

def matrixMultiply(mat1, mat2):
    if(len(mat1[0])==len(mat2)):
        ret = []
        for i in range(len(mat1)):
            tempmat = []
            for j in range(len(mat2[0])):
                tempsum = 0
                for k in range(len(mat2)):
                    tempsum += mat1[i][k]*mat2[k][j]
                tempmat.append(tempsum)
            ret.append(tempmat)
        return ret
    else:
        print("Dimensions do not match in matrix multiplication!")

def toMatrix(vector):
    newmat = []
    newmat.append(vector)
    return newmat

class neuralNetwork:
    def __init__(self, nnmap):
        self.networkMap = nnmap
        self.number_layers = len(nnmap)
        totaln = 0
        for i in nnmap:
            totaln += i
        self.totalNeuron = totaln
        self.bias = 1
        self.weight = []
        self.Output = []
        self.layerErrorList = []
        self.activationList = []

    def weightMat(self):
        count = 0
        for k in self.networkMap:
            if(count!=0):
                tempweight = [[0 for i in range(pre + 1)] for j in range(k)] #first rows then columns
                for i in range(k):
                    for j in range(pre + 1):
                        tempweight[i][j] = random.uniform(-1, 1)
                self.weight.append(tempweight)
            pre = k
            count += 1

    def feed(self, input_list):
        self.weightMat()
        input_list.append(self.bias)
        for i in self.weight:
            templist = []
            for j in i:
                rowsum = 0
                for k in range(len(j)):
                    rowsum += j[k]*input_list[k]
                templist.append(sigmoid(rowsum))
            templist.append(self.bias)
            input_list = templist
            self.activationList.append(input_list)
        input_list = input_list[:-1]
        self.Output = input_list

    def prediction(self):
        max = self.Output[0]
        for i in self.Output:
            if(i > max):
                max = i
        return max

    def backPropagate(self, expectedOutput):
        templayerE = []
        for i in range(len(expectedOutput)):
            templayerE.append(expectedOutput[i]-self.Output[i])
        self.layerErrorList.append(templayerE)
        tempmap1 = self.networkMap[:-1]
        tempmap = []
        for i in range(len(tempmap1)):
            i+=1
            tempmap.append(tempmap1[-i])
        tempmap = tempmap[:-1]
        revweight = []
        for i in range(len(self.weight)):
            i+=1
            revweight.append(self.weight[-i])
        for i in range(len(tempmap)):
            templayererror = []
            for j in range(tempmap[i]):
                error = 0
                for k in range(len(self.layerErrorList[i])):
                    summ = 0
                    for l in revweight[i][k]:
                        summ += l
                    error += (revweight[i][k][j]/summ)*self.layerErrorList[i][k]
                templayererror.append(error)
            self.layerErrorList.append(templayererror)
        layer = []
        for i in range(len(self.layerErrorList)):
            i+=1
            layer.append(self.layerErrorList[-i])
        self.layerErrorList = layer

    '''def trainit(self, inputDataset, outputDataset):
        for i in range(len(inputDataset)):
            self.getOutput(inputDataset[i])
            self.backPropagate(outputDataset[i])
            newWeight = []
            for j in range(len(self.layerErrorList)):
                for k in range(len(self.layerErrorList[j])):
                    if(j==0):
                        es = sign(self.layerErrorList[j][k])
                        signmat = '''
    def recall(self, outputlist):
        outputlist = transpose(toMatrix(outputlist))
        revweight = []
        for i in range(len(self.weight)):
            i+=1
            revweight.append(self.weight[-i])
        for i in revweight:
            for j in range(len(i)):
                temp = i[j]
                temp = temp[:-1]
                i[j] = temp
            outputlist = matrixMultiply(inverse(i), outputlist)
        return outputlist

# end of library

map = [5, 5, 5]
nn1 = neuralNetwork(map)
inputs = [0.1, 0.8, 0.4, 0.9, 0.3]
nn1.feed(inputs)
#print(nn1.Output)
print(nn1.activationList)
print('Predicted value is: ',nn1.prediction())
exp = [1,0,1,1,0]
input1 = [[0.1],[0.8],[0.4],[0.9],[0.3],[1]]
#nn1.backPropagate(exp)
#print(nn1.layerErrorList)
#print(nn1.recall(nn1.Output))
print(nn1.weight[0])
print(sigmoid_matrix(matrixMultiply(nn1.weight[0], input1)))
