# -*- coding: utf-8 -*-
import copy
import random

import numpy as np
import geatpy as ea
import torch.nn.functional as F
import torch.optim as optim
import torch
from utils import accuracy,getInfluNode
import math
import time
import dgl
import scipy.sparse as ssp
from sklearn.metrics import roc_auc_score
from models.GCN import GCN
import pandas as pd

class RepairSubgraphCOV(ea.Problem):
    def __init__(self,rawGraph,adj,model,features,labels, idx_train, idx_val, idx_test,sens,idx_sens_train,args):
        name = 'RepairSubgraphCOV'
        M = 3
        self.rawG=rawGraph
        self.sparseAdj=adj
        self.denAdj=adj.toarray()
        self.args=args
        self.features=features
        self.labels=labels
        self.idx_train=idx_train
        self.idx_val=idx_val
        self.idx_test=idx_test
        self.sens=sens
        self.idx_sens_train=idx_sens_train
        self.rawModel=model
        self.M=M
        maxormins = [1] * M
        Dim = int((rawGraph.num_nodes() * rawGraph.num_nodes() - rawGraph.num_nodes()) / 2)
        varTypes = [1] * Dim
        lb = [0] * Dim
        ub = [1] * Dim
        lbin = [1] * Dim
        ubin = [1] * Dim
        self.rawGNN_SP,self.rawGNN_EO,_=self.trainModel(model)
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)

    def fair_metric(self,output,idx):
        labels=self.labels
        sens=self.sens
        val_y = labels[idx].cpu().numpy()
        idx_s0 = sens.cpu().numpy()[idx.cpu().numpy()]==0
        idx_s1 = sens.cpu().numpy()[idx.cpu().numpy()]==1
        idx_s0_y1 = np.bitwise_and(idx_s0,val_y==1)
        idx_s1_y1 = np.bitwise_and(idx_s1,val_y==1)
        pred_y = (output[idx].squeeze()>0).type_as(labels).cpu().numpy()
        parity = abs(sum(pred_y[idx_s0])/sum(idx_s0)-sum(pred_y[idx_s1])/sum(idx_s1))
        equality = abs(sum(pred_y[idx_s0_y1])/sum(idx_s0_y1)-sum(pred_y[idx_s1_y1])/sum(idx_s1_y1))
        return parity,equality



    def updatePartialModel(self,G,newAdj,directInfluNode):
        args=self.args
        model=self.model
        H=self.H.clone()
        maxH,_=H.max(0)
        minH,_=H.min(0)
        features=self.repairFeatures(G).detach()
        labels=self.labels
        idx_test=self.idx_test
        idx_val=self.idx_val
        allInfluNode=getInfluNode(directInfluNode,newAdj,H)
        allInfluNode=list(set(allInfluNode))
        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(3):
            model.train()
            optimizer.zero_grad()
            output = model(G, features)
            loss_train=F.binary_cross_entropy_with_logits(output[allInfluNode],H[allInfluNode].detach().float())
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                model.eval()
                output = model(G, features)

            parity_val, equality_val = self.fair_metric(output,idx_val)
            roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
        return parity_val,equality_val,roc_val



    def trainModel(self,model):
        args=self.args
        features=self.features
        idx_train=self.idx_train
        labels=self.labels
        idx_val=self.idx_val
        optimizer = optim.Adam(model.parameters(),lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            output = model(self.rawG, features)
            loss_train = F.binary_cross_entropy_with_logits(output[idx_train], labels[idx_train].unsqueeze(1).float())
            loss_train.backward()
            optimizer.step()
            if not args.fastmode:
                model.eval()
                output = model(self.rawG, features)
            parity_val, equality_val = self.fair_metric(output,idx_val)
            roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())

        self.model=model
        self.H=output
        return parity_val,equality_val,roc_val


    def aimFunc(self, pop):
        pop.ObjV=np.zeros((pop.Chrom.shape[0],self.M))
        for i in range(pop.Chrom.shape[0]):
            sub=pop.Chrom[i,:]
            dAdj=self.denAdj
            directInfluNode=[]
            numOfRemove=0
            numofAdd=0
            selectedEdge=np.where(sub==1)[0]
            for sEdge in selectedEdge:
                row=math.floor((-0.5+math.sqrt(0.25-4*-0.5*sEdge))/1)+1
                col=int(sEdge-((row-1)*(row-1)-((row-1)*(row-1)-(row-1))/2))
                if dAdj[row,col]==1:
                    dAdj[row,col]=0
                    numOfRemove=numOfRemove+1
                if dAdj[row,col]==0:
                    dAdj[row,col]=1
                    numofAdd=numofAdd+1
                directInfluNode.append(row)
                directInfluNode.append(col)
            subnode=list(set(directInfluNode))
            subAdj=dAdj[subnode,:]
            subAdj=subAdj[:,subnode]
            compact=(len(np.where(subAdj!=0)[0]))/(len(subnode)+1)
            G=dgl.from_scipy(ssp.csr_matrix(dAdj))
            G = dgl.add_self_loop(G)
            G = G.to('cuda:0')
            directInfluNode=list(set(directInfluNode))
            sp,eo,roc=self.updatePartialModel(G,dAdj,directInfluNode)
            pop.ObjV[i,:]=[sp,-roc,compact]





