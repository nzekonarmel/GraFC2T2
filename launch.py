# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:41:45 2019

@author: NZEKON
"""
import framework as fw

# ---->  input data

# file that contains link stream data {(u, i, r, c, t)}
input_dataset_file = "ciao-data.txt" 
separator_df = ";"   # separator of elements in a row of the dataset file
pos_u = 0   # position of user u in a row [the first position is 0 (zero)]
pos_i = 1   # position of item i 
pos_r = 2   # position of rating r 
pos_c = 3   # position of content-based feature c 
pos_t = 4   # position of timestamp t

# file that contains explicit trust network
# This is optional if you don't want to use explicit trust in GraFC2T2
input_trust_network = "ciao-trust-network.txt" 
separator_tn = ";"   # separator of elements in a row of the trust network file
pos_user = 0        # position of user u who trust another
pos_trusted = 1     # position of the trusted user

 
# ---->  evaluation metrics

# you can add any metric in the form prec@N (Precision), recall@N, map@N, hr@N (Hit Ratio)
# with N in {1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100}
eval_metric = ["prec@10", "recall@10", "map@10", "hr@10"]   

# ---->  parameters of experiment

# the time is divided in time slices of equal length
number_of_time_slices = 8 # this value >= 2


"""

 It is possible to configure several recommender graphs
 In this example, two recommender graphs are configurated 

 ---->  recommender graph construction

 graph : basic graph {"BIP", "STG", "LSG"}
 content : content-based method {"NA", "CI", "CIU"}
 time : time-weight function {"NA", "EDF", "LDF"}
 trust : trust method in PageRank {"NA", "ET", "IT"}

 ---->  recommender graph parameters setting

 Delta : STG session duration    {7,30,90,180,365,540,730} in days
 Beta : STG long-term preference    {0.1, 0.3, 0.5, 0.7, 0.9}  
 T0 : half life of Exponential Decay Function (EDF) and Logistic Decay Function (LDF) {7,30,90,180,365,540,730} in days  
 K : decay slope of LDF    {0.1, 0.5, 1, 5, 10, 50, 100}
 Gamma : influence of trusted users    {0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9}         
 Alpha : damping factor for PageRank    {0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9}

"""
rs1_param = {
    "graph":"BIP", "content":"CIU", "time":"EDF", "trust":"IT", 
    "delta":0, "beta":0, "t0":180, "k":0, "gamma":0.9, "alpha":0.9
}

rs2_param = {
    "graph":"STG", "content":"CIU", "time":"LDF", "trust":"IT", 
    "delta":540, "beta":0.1, "t0":365, "k":10, "gamma":0.7, "alpha":0.9
}

rs_list = [rs1_param, rs2_param] # list of configurated recsys 


# ---->  Read data
import csv

linkstream = []
with open(input_dataset_file,'rb') as input_file:
    input_file = csv.reader(input_file, delimiter=separator_df)
    for line in input_file:
        if len(line) >= 5:
            t,u,i,c,r = int(line[pos_t]), line[pos_u], line[pos_i], line[pos_c], float(line[pos_r]) 
            linkstream.append( (t, u, i, c, r) )
linkstream = sorted(linkstream, key=lambda tup: tup[0])

trust_network = []
if input_trust_network!=None or input_trust_network!="":
    with open(input_trust_network,'rb') as input_file:
        input_file = csv.reader(input_file, delimiter=separator_tn)
        for line in input_file:
            if len(line) >= 2:
                u,trusted = line[pos_user], line[pos_trusted]
                trust_network.append( (u, trusted) )

fw.main(rs_list, linkstream, trust_network, eval_metric, number_of_time_slices)

