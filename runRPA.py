#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 17:25:52 2020

@author: nvthaomy
"""
import RPA as RPA

RPA = RPA.RPA(5,5)
RPA.Setchain('DGC')
RPA.SetDOP([6,6,1,1,1])
RPA.Setcharge([-1,1,1,-1,0])
RPA.SetC([0.25931602946650645,0.25826444797657155,0.49576946342269224,0.49471788193, 31.421473668958015])
RPA.Setabead([0.45,0.45,0.31,0.31,0.31])
RPA.Setu0([[2.488114151573582, 3.0118464709458257,0.025693680984791637,2.138921461589263,1.1332953245877369],
           [3.0118464709458257, 6.402345482203565,1.6866826606347285,1.4729241456705382,1.3459231140580075],
           [0.025693680984791637,1.6866826606347285,0.7948861837682706,0.013269823077509207,0.013269159533271794],
           [2.138921461589263,1.4729241456705382,0.013269823077509207,1.921093276162326,0.6808096584719776],
           [1.1332953245877369,1.3459231140580075,0.013269159533271794,0.6808096584719776,0.44984318031275466]])
#RPA.Setu0([[1., 2., 1. ],
#           [2., 1., 0.5],
#           [1.,0.5,0.5]])
RPA.SetlB(9.349379737083224)
RPA.Setb([0.14256642675433792,0.1475274599800548,1.,1.,1.])
RPA.SetV(6.)
RPA.Setkmin(1e-5)
RPA.Setkmax(100)
RPA.Setnk(1000)

RPA.Initialize()

print(RPA.mu(0))
print(RPA.mu(1))
print(RPA.mu(2))
print(RPA.mu(3))
print(RPA.mu(4))
print(RPA.P())