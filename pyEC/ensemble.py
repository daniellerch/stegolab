#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import tempfile
import shutil
import numpy
from scipy.io import savemat, loadmat

MATLAB="/usr/local/MATLAB/R2013a/bin/matlab -nodesktop -nojvm -nosplash -r"


class EC:


    def fit(self, X, y):
        
        self.__tmpdir=tempfile.mkdtemp()

        y=numpy.array(y)
        Xc=X[y==0]
        Xs=X[y==1]
        
        if len(Xc)>len(Xs):
            Xs=Xs[:len(Xc)]

        if len(Xs)>len(Xc):
            Xc=Xc[:len(Xs)]

        pcover=self.__tmpdir+"/F_train_cover.mat"
        savemat(pcover, mdict={'F': numpy.array(Xc)}, oned_as='column')

        pstego=self.__tmpdir+"/F_train_stego.mat"
        savemat(pstego, mdict={'F': numpy.array(Xs)}, oned_as='column')

        pclf=self.__tmpdir+"/clf.mat"
    
        cwd=os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        output=os.popen(MATLAB+" \" ensemble_fit('"+
            pcover+"', '"+pstego+"', '"+pclf+"');exit \"").read()
        os.chdir(cwd)

        self.__mat_clf=loadmat(pclf)
        shutil.rmtree(self.__tmpdir)

    def predict_proba(self, X):

        self.__tmpdir=tempfile.mkdtemp()

        prob=[]

        path=self.__tmpdir+"/F_test.mat"
        savemat(path, mdict={'F': numpy.array(X)}, oned_as='column')

        pclf=self.__tmpdir+"/clf.mat"
        savemat(pclf, self.__mat_clf)

        pvotes=self.__tmpdir+"/votes.txt"

        cwd=os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        output=os.popen(MATLAB+" \" ensemble_predict('"+
            pclf+"', '"+path+"', '"+pvotes+"');exit \"").read()
        os.chdir(cwd)

        with open(pvotes, 'r') as f:
            lines=f.readlines()
        f.close()

        shutil.rmtree(self.__tmpdir)

        for l in lines:
            votes=(1+float(l)/500)/2
            prob.append( [1-votes, votes] )

        return prob


    def predict(self, X):
        results=[]
        proba=self.predict_proba(X)
        for p in proba:
            if p[0]>=0.5:
                results.append(0)
            else:
                results.append(1)
        return numpy.array(results)

    def score(self, X, y):
        Z=self.predict(X)
        result=numpy.count_nonzero(Z==y)
        return round(float(result)/len(y), 2)


    def save(self, path):
        savemat(path, self.__mat_clf)

    def load(self, path):
        self.__mat_clf=loadmat(path)

