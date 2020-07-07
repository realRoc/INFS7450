# -*- coding: utf-8 -*-
"""
Created on Tue Jul  7 13:07:44 2020

@author: simon
"""

def readData(filename):
    data = []
    with open(filename) as f:
        for line in f:
         data.append(line.strip().split())
    return data

GroundTruth = readData('groundtruth.txt')
result = readData('results.txt')

count = 0
for pair in result:
    if pair in GroundTruth:
        count += 1

print('The prediction\'s accuracy is:', round(count/len(GroundTruth)*100,2), '%')