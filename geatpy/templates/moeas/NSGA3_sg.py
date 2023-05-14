# -*- coding: utf-8 -*-
import math

import numpy as np

import geatpy as ea
from sys import path as paths
from os import path as path
import random
paths.append(path.split(path.split(path.realpath(__file__))[0])[0])


class NSGA3_sg(ea.MoeaAlgorithm):

    def __init__(self, problem, population):
        ea.MoeaAlgorithm.__init__(self, problem, population)
        self.name = 'NSGA3_sg'
        if self.problem.M < 10:
            self.ndSort = ea.ndsortESS
        else:
            self.ndSort = ea.ndsortTNS
        self.selFunc = 'urs'
        if population.Encoding == 'P':
            self.recOper = ea.Xovpmx(XOVR=1)
            self.mutOper = ea.Mutinv(Pm=1)
        elif population.Encoding == 'BG':
            self.recOper = ea.Xovud(XOVR=1)
            self.mutOper = ea.Mutbin(Pm=None)

        elif population.Encoding == 'RI':
            self.recOper = ea.Recsbx(XOVR=1, n=20)
            self.mutOper = ea.Mutpolyn(Pm=1 / self.problem.Dim, DisI=20)
        else:
            raise RuntimeError('Error')

    def reinsertion(self, population, offspring, NUM, uniformPoint):


        population = population + offspring
        [levels, criLevel] = self.ndSort(population.ObjV, NUM, None, population.CV,
                                         self.problem.maxormins)
        chooseFlag = ea.refselect(population.ObjV, levels, criLevel, NUM, uniformPoint,
                                  self.problem.maxormins)
        return population[chooseFlag]


    def subCrossOper(self,OldChrom,Pcross=0.8,randSelectedRatio=0.5):
        numOfCross=int(OldChrom.shape[0]*Pcross/2)
        popsize=int(OldChrom.shape[0])
        offChrom=OldChrom.copy()
        for i in range(numOfCross):
            selectedIndi=random.sample(range(0,popsize),2)
            parent1=OldChrom[selectedIndi[0],:]
            parent2=OldChrom[selectedIndi[1],:]
            commonEdge=parent1 & parent2
            parent1EdgeIndex=np.where(parent1==1)[0]
            parent2EdgeIndex=np.where(parent2==1)[0]
            np.random.shuffle(parent1EdgeIndex)
            np.random.shuffle(parent2EdgeIndex)
            parent1SelectedEdgeIndex=parent1EdgeIndex[0:int(len(parent1EdgeIndex)*randSelectedRatio)]
            parent2SelectedEdgeIndex=parent2EdgeIndex[0:int(len(parent2EdgeIndex)*randSelectedRatio)]
            parent1SelectedEdge=np.zeros((OldChrom.shape[1]))
            parent1SelectedEdge[parent1SelectedEdgeIndex]=1
            parent2SelectedEdge=np.zeros((OldChrom.shape[1]))
            parent2SelectedEdge[parent2SelectedEdgeIndex]=1
            parent1SelectedEdge=parent1SelectedEdge.astype(int)
            parent2SelectedEdge=parent2SelectedEdge.astype(int)
            offChrom[selectedIndi[0],:]=(commonEdge | parent1SelectedEdge)
            offChrom[selectedIndi[1],:]=(commonEdge | parent2SelectedEdge)
        return offChrom

    def subMutOper(self,OldChrom,rawAdj,Pmut=0.8,randSelectedRatio=0.5):
        numOfMut=int(OldChrom.shape[0]*Pmut)
        popsize=int(OldChrom.shape[0])
        offChrom=OldChrom.copy()
        numOfNode=rawAdj.shape[0]
        for i in range(numOfMut):
            selectedIndi=random.sample(range(0,popsize),1)
            parent1=OldChrom[selectedIndi[0],:]
            expandList=np.where(parent1==1)[0]
            np.random.shuffle(expandList)
            expandIndex=expandList[0]
            row=math.floor((-0.5+math.sqrt(0.25-4*-0.5*expandList[0]))/1)+1
            col=expandList[0]-((row-1)*(row-1)-((row-1)*(row-1)-(row-1))/2)
            canSelectList=rawAdj[row,:]
            canSelectList=np.where(canSelectList>0)[0].astype(int)
            np.random.shuffle(canSelectList)
            for node in canSelectList:
                Iindex=row
                Jindex=node
                if Jindex>Iindex:
                    temp=Jindex
                    Jindex=Iindex
                    Iindex=temp
                canSetOneChroIndex=int(Iindex*(Iindex+1)/2+Jindex-Iindex)
                if canSetOneChroIndex>=len(parent1):
                    continue
                if parent1[canSetOneChroIndex] !=0:
                    offChrom[selectedIndi[0],canSetOneChroIndex]=1
                    break
        return offChrom




    def run(self, rawAdj,prophetPop=None):
        population =prophetPop
        self.initialization()
        uniformPoint, NIND = ea.crtup(self.problem.M, population.sizes)
        evoI=1
        while self.terminated(population) == False:
            offspring = population[ea.selecting(self.selFunc, population.sizes, NIND)]
            offspring.Chrom=self.subCrossOper(offspring.Chrom,0.8)
            offspring.Chrom=self.subMutOper(offspring.Chrom,rawAdj,0.8)
            self.call_aimFunc(offspring,evoI)
            evoI=evoI+1
            population = self.reinsertion(population, offspring, NIND, uniformPoint)
        return self.finishing(population)
