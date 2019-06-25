from prettytable import PrettyTable
import math
import time
import datetime
import inspect
import matplotlib.pyplot as plt
import numpy as np
import random

DEBUG = True

def log(s):
    if DEBUG:
        print (s)

def function1(x,y):
    return -math.cos(x)*math.cos(y)*math.exp( -pow(x-math.pi,2) -pow(y-math.pi,2) )

def function2(x,y):
    return x**2 + y**2

#Funciones de paper
def f1(x,y):
    return x * math.sin(4*x) + 1.1*y*math.sin(2*y)


 
class FOA:
    def __init__(self,function,lowlim,highlim,lifeTime,LSC,GSC,transferRate,areaLimit,forestSize,minimize,generations):
        self.lifeTime     = lifeTime
        self.LSC          = LSC
        self.GSC          = GSC
        self.transferRate = transferRate
        self.areaLimit    = areaLimit
        self.forestSize   = forestSize
        
        self.function     = function
        self.funcArgs     = len(inspect.getfullargspec(self.function)[0])
        self.lowlim       = lowlim
        self.highlim      = highlim

        self.diffX        = (self.highlim*0.15)  #Valor a añadir a la variable elegida (Local seeding)   

        self.minimize     = minimize
        self.best         = np.array(["Aún","No existe"])
        self.infs         = ["-inf","inf"]
        self.bestFit      = float( self.infs[self.minimize] )
        self.generations  = generations

        self.candidates   = np.array([]).reshape((0,self.funcArgs+1))
        
        self.generateInitialForest()
        self.forestFitness = self.fitness(self.forest)
        self.printTable("Población Inicial",self.forest,self.forestFitness)

        plt.xlim(self.lowlim,self.highlim)
        plt.ylim(self.lowlim,self.highlim)
        
        for i in range(self.generations):
            log("\n"+"+"*70); log("GENERACIÓN {} DE {}:".format(i+1,self.generations))
            #Local seeding
            self.plotTrees(self.forest,'ro',6)
            self.localSeeding()
            self.printTable("Población (Local Seeding)",self.forest,self.forestFitness)
        
            #Population limiting
            self.populationLimiting()
            self.printTable("Población (Population Limiting)",self.forest,self.forestFitness)
            self.printTable("Población Candidatos",self.candidates)
            #Global seeding (mutation)
            self.globalSeeding()
            self.printTable("Población (Global Seeding)",self.forest,self.forestFitness)
            #Update best so far
            self.updateBest()
            self.plotTrees([self.best],'g^',15)
            plt.clf()

        print("El mejor es {} ({})".format( self.best[1:], self.bestFit ) )

    def generateInitialForest(self):
        self.forest    = np.array( [ [0]+[ np.random.uniform(self.lowlim,self.highlim)\
                                 for i in range (self.funcArgs) ]\
                                 for j in range (self.forestSize) ] )

    def fitness(self,trees):
        return np.array([ self.function(*i[1:])  for i in trees ])

    def localSeeding(self):
        log("Creación de nuevos arboles")
        newTrees = []
        for i in self.forest:
            if( i[0] == 0): #Verificamos si es un arbol joven (edad 0)
                for j in range(self.LSC):
                    newTree = np.copy(i)
                    randVar = np.random.randint(1,self.funcArgs+1)      #Variable a editar
                    randVal = np.random.uniform(-self.diffX,self.diffX) #Valor a añadir
                    log("{} => {} + {} = {}".format(i[1:],i[randVar],randVal,i[randVar]+randVal))
                    newTree[randVar] += randVal
                    newTree[randVar] = self.checkBoundaries(newTree[randVar])
                    newTrees.append(newTree)    
                
        log("Incrementando edad a todos los arboles antiguos")
        #Aumentamos la edad de los arboles
        for i in self.forest:
            i[0] += 1
        #Añadimos los nuevos generados al bosque
        self.plotTrees(newTrees,'bo',6)
        self.forest = np.concatenate( ( self.forest, np.array(newTrees) ) )
        self.forestFitness = np.concatenate( (self.forestFitness, self.fitness(newTrees)  ) )


    def populationLimiting(self):
        #Eliminamos los que son mayores al lifeTime
        oldTreesIndex, keptTreesIndex = [],[]
        for i in range ( len(self.forest) ):
            if (self.forest[i][0] >= self.lifeTime):
                oldTreesIndex.append(i)
            else:
                keptTreesIndex.append(i)
        self.candidates    = np.concatenate( ( self.candidates, self.forest[oldTreesIndex] ) )
        self.forest        = self.forest[keptTreesIndex]
        self.forestFitness = self.forestFitness[keptTreesIndex]
        #Ordenamos el bosque de acuerdo al fitness
        sorted             = self.forestFitness.argsort()
        #Eliminamos los arboles extra de acuerdo a areaLimit y los añadimos a candidatos
        kept, deleted      = self.keptAndDeleted(sorted,self.areaLimit)
        self.candidates    = np.concatenate( ( self.candidates, self.forest[deleted] ) )
        self.forest        = self.forest[kept]
        self.forestFitness = self.forestFitness[kept]

    def globalSeeding(self):
        #Calculamos el transferRate% del total de candidatos
        chosenAmmount = math.ceil((self.transferRate/100)*len( self.candidates ))
        log("\nSeleccionamos {} elementos ({}% del total de candidatos)".format(chosenAmmount,self.transferRate))
        #Dos opciones (La del paper es la Op2)
        #Seleccionamos el transferRate% de los mejores candidatos. (Op1) 
        # self.candidatesFitness = self.fitness(self.candidates)
        # sortedCandidates = self.forestFitness.argsort()
        # kept, deleted      = self.keptAndDeleted(sortedCandidates,chosenAmmount)
        # chosenCandidates = self.candidates[kept]
        
        #Seleccionamos aleatoriamente transferRate% del total de candidatos (Op2)
        np.random.shuffle(self.candidates)
        chosenCandidates = self.candidates[:chosenAmmount]

        newGenerated = []
        for i in chosenCandidates:
            candidate    = np.copy(i)
            candidate[0] = 0
            for j in range(self.GSC):
                randVar      = np.random.randint(1,self.funcArgs+1)         #Variable a editar
                randVal      = np.random.uniform(-self.lowlim,self.highlim) #Valor con el cual reemplazar
                candidate[randVar] = randVal
            newGenerated.append(candidate)
        #Añadimos los nuevos generados al bosque
        newGenerated        = np.array(newGenerated)
        newGeneratedFitness = self.fitness(newGenerated)
        self.forest         = np.concatenate( ( self.forest, newGenerated ) )
        self.forestFitness  = np.concatenate( ( self.forestFitness, newGeneratedFitness ) )
        #Vaciamos los candidatos
        self.candidates     = np.array([]).reshape((0,self.funcArgs+1))

    def updateBest(self):
        #Ordenar valores de acuerdo al fitness

        #Actualizar mejor valor y mejor arbol edad 0
        localBest = self.localBest(self.forestFitness)
        better    = self.isBetterThan(self.forestFitness[localBest], self.bestFit  )
        if( better ):
            log("El mejor {} ({}) se actualiza a {} ({})".format(\
                    self.best[1:],self.bestFit,self.forest[localBest][1:],self.forestFitness[localBest] ))
            #Reiniciamos la edad del mejor.
            self.forest[localBest][0] = 0
            #Actualizamos el mejor
            self.best                 = self.forest[localBest]
            self.bestFit              = self.forestFitness[localBest]
            
        else:
            log("El mejor {} ({}) se mantiene".format(self.best[1:],self.bestFit))
            same = True
            for i in range(1, len(self.forest[localBest]) ):
                if( self.best[i] != self.forest[localBest][i] ):
                    same = False
            if(same):
                log("El mejor se encuentra en el bosque, reiniciando edad.")
                self.forest[localBest][0] = 0
                self.best = self.forest[localBest]
            

        
    #------------------------------FUNCIONES DE SOPORTE----------------------------------------#

    def keptAndDeleted(self,sorted,limit):
        if(self.minimize):
            return sorted[:limit],sorted[limit:]
        else:
            return sorted[-limit:],sorted[:-limit]

    def localBest(self,population):
        if self.minimize:
            return population.argmin()
        else:
            return population.argmax()

    def isBetterThan(self,val1,val2):
        if self.minimize:
            if val1 < val2:
                return True
        else:
            if val1 > val2:
                return True
        return False

    def makeAxis(self,trees):
        x,y = [],[]
        for i in trees:
            x.append(i[1])
            y.append(i[2])
        return x,y

    def checkBoundaries(self,val):
        if(val > self.highlim):
            return self.highlim
        elif( val < self.lowlim ):
            return self.lowlim
        
        return val
            

    def plotTrees(self,trees,options,markersiz):
        plt.xlim(self.lowlim,self.highlim)
        plt.ylim(self.lowlim,self.highlim)
        plt.plot(*self.makeAxis(trees),options,markersize=markersiz)
        plt.draw()
        plt.pause(0.005)

    #Dependiendo de la cantidad de soluciones,
    #haremos cabeceras de la tabla.
    def makeFields(self):
        fields = []
        for i in range(0,self.funcArgs):
            field = "x"+str(i+1)
            fields.append(field)
        return fields

    def printTable(self,title,pop,fit=[]):
        log("\n"+title)
        pops = lambda x: list(x) if (len(x)>1)  else list([x])
        table = PrettyTable()
        if( len(fit) > 0 ):
            table.field_names = ["#"] + self.makeFields() + ["Fitness","Edad"]
            for i in range(len(pop)):
                vals = [i+1]+pops(pop[i][1:]) + [fit[i],int(pop[i][0])]
                table.add_row( vals )
        else:
            table.field_names = ["#"] + self.makeFields() + ["Edad"]
            for i in range(len(pop)):
                vals = [i+1]+pops(pop[i][1:]) + [int(pop[i][0])]
                table.add_row( vals )     
       
        log(table.get_string(header=True, border=True))


if __name__ == '__main__':
    print("\nCOMIENZO PROCESO: ", datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    comienzo = time.time()
    foa   = FOA(function1,lowlim = -10,highlim = 10,\
                lifeTime = 4, LSC = 2, GSC = 1, transferRate = 10, areaLimit = 30, forestSize = 30,\
                minimize = True, generations = 20)
    final    = time.time()
    print("\nFIN DEL PROCESO", datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S"))
    print("Tiempo:",final-comienzo)
