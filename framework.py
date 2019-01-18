# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 22:27:24 2018

@author: NZEKON
"""
#
###############################################################################
###############################################################################
# 
# UTILS
#
###############################################################################
#
import os
import shutil
import time
import os.path
import numpy as np

workdir = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(workdir,"out")

# ------------------------------------------------------------------------------
# File input and output in the out_dir
# ------------------------------------------------------------------------------
class Out (object):
    outDir = out
    
    def __init__(self, file, newdir=None, topen="w"):
        self.filename = ""
        if newdir != None:
            self.filename = os.path.join(Out.outDir,newdir)
            mkdir(self.filename)
            self.filename = os.path.join(self.filename,file+".txt")
        else:    
            self.filename = os.path.join(Out.outDir,file+".txt")
        self.filetmp = safeOpen(self.filename, topen)
    
    def write(self, text):
        self.filetmp.write(str(text)+"\n")
        
    def writewt(self, text): # write with time
        instant = time.strftime('%d/%m/%y %H:%M:%S',time.localtime()) 
        ligne = instant+" ; "+str(text)+"\n"
        self.filetmp.write(str(ligne))
        
    def close(self):
        self.filetmp.close()

    @staticmethod    
    def copy(srcdir, dstdir, filename):    
        outsrcdir = os.path.join(Out.outDir,srcdir) 
        outsrcfile = os.path.join(outsrcdir,filename)
        outdstdir = os.path.join(Out.outDir,dstdir)
        os.path.isdir(outsrcdir)
        os.path.isdir(outdstdir)
        if not os.path.exists(outdstdir):
            os.makedirs(outdstdir)
        shutil.copy2(outsrcfile, outdstdir)
    
    @staticmethod
    def dataDistCdfCcdf(tableau):
        tab = np.array(tableau, float) 
        tab.sort()
        minv = tab[0]
        maxv = tab[-1:][0]
        dist = {}
        i = 0
        while i < len(tab):
            if not tab[i] in dist.keys():
                dist[tab[i]] = 1
            else:
                dist[tab[i]] += 1
            i += 1
        nb = 0
        maxx = len(tableau)
        valDist = np.array(dist.keys())
        valDist.sort()
        
        # CDF
        cdf = {}
        for i in range(0,len(valDist)):
            key = valDist[i]
            nb += dist[key]
            cdf[key] = (100.0 * nb)/maxx #nb

        for key in dist:
            dist[key] = (100.0*dist[key])/maxx

        # CCDF
        ccdf = {}
        i = 0
        while i < len(valDist):
            key = valDist[i]
            ccdf[key] = (100.0 * nb)/maxx #nb
            nb -= dist[key]
            i += 1
        
        return dist,cdf,ccdf,minv,maxv

    @staticmethod
    def distAndCcdf(tableau):
        tab = np.array(tableau, float) 
        tab.sort()
        minv = tab[0]
        maxv = tab[-1:][0]
        dist = {}
        i = 0
        while i < len(tab):
            if not tab[i] in dist.keys():
                dist[tab[i]] = 1
            else:
                dist[tab[i]] += 1
            i += 1
        nb = len(tab)
        maxx = nb
        valDist = np.array(dist.keys())
        valDist.sort()
        ccdf = {}
        i = 0
        while i < len(valDist):
            key = valDist[i]
            ccdf[key] = (100.0 * nb)/maxx #nb
            nb -= dist[key]
            i += 1
        return dist,ccdf,minv,maxv    
        
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exc: # python > 2.5
        if exc.errno == os.errno.EEXIST and os.path.isdir(path):        
            pass
        else:
            raise

def safeOpen(path, topen="w"):
    mkdir(os.path.dirname(path))
    return open(path, topen)  
#
###############################################################################

###############################################################################
# 
# RECSYSBASIC
#
###############################################################################
#
#  EVALUATION OF RECOMMENDER SYSTEMS
#
metrics = ["hr","prec","recall","map","mrr","f0.5","f1","f2"]
tops_n = [1,2,3,5,10,15,20,30,40,50,100]
eval_metrics = [
    "hr@1","hr@2","hr@3","hr@5","hr@10","hr@15","hr@20","hr@30","hr@40","hr@50","hr@100",
    "prec@1","prec@2","prec@3","prec@5","prec@10","prec@15","prec@20","prec@30","prec@40","prec@50","prec@100",  
    "recall@1","recall@2","recall@3","recall@5","recall@10","recall@15","recall@20","recall@30","recall@40","recall@50","recall@100", 
    "map@1","map@2","map@3","map@5","map@10","map@15","map@20","map@30","map@40","map@50","map@100", 
    "mrr@1","mrr@2","mrr@3","mrr@5","mrr@10","mrr@15","mrr@20","mrr@30","mrr@40","mrr@50","mrr@100",
    "f0.5@1","f0.5@2","f0.5@3","f0.5@5","f0.5@10","f0.5@15","f0.5@20","f0.5@30","f0.5@40","f0.5@50","f0.5@100",
    "f1@1","f1@2","f1@3","f1@5","f1@10","f1@15","f1@20","f1@30","f1@40","f1@50","f1@100",
    "f2@1","f2@2","f2@3","f2@5","f2@10","f2@15","f2@20","f2@30","f2@40","f2@50","f2@100"
]

class Evaluation (object):
    
    @staticmethod
    def evaluation_metric_list():
        return eval_metrics
    
    def __init__ (self, links_to_rec, rec_links):
        
        #if len(rec_links) > 0:
        #    print "rec_links",rec_links[rec_links.keys()[0]]
        #    print "links_to_rec",links_to_rec[links_to_rec.keys()[0]]
        if len(links_to_rec) < 0: raise ValueError('Evaluation : links_to_rec bad value!')
        if len(rec_links) < 0: raise ValueError('Evaluation : rec_links bad value!')
        self.links_to_rec = links_to_rec
        self.rec_links = rec_links
        self.rec_links_binary = {}
        self.result_values = {}
        self.result_weights = {}
        self.evaluation_metrics = eval_metrics
        
    def get_evaluation_metrics(self):
        return self.evaluation_metrics
    
    def get_result_values(self):
        return self.result_values
    
    def get_result_weights(self):
        return self.result_weights
        
    def compute_evaluation_results(self):
        # transform all recommendation list to binary list
        for u in self.links_to_rec.keys():
            self.rec_links_binary[u] = []
            #print u,"-------"
            #print list(self.links_to_rec[u])[:5]
            #print list(self.rec_links[u])[:5]
            for i in self.rec_links[u]:
                if i in self.links_to_rec[u]:
                    self.rec_links_binary[u].append(1)
                else:
                    self.rec_links_binary[u].append(0)
        
        # return all evaluation
        eval_results,eval_result_weights = {},{}
        eval_results["hr"],eval_result_weights["hr"] = self._get_hit_ratio()
        eval_results["prec"],eval_result_weights["prec"] = self._get_precision()
        eval_results["recall"],eval_result_weights["recall"] = self._get_recall()
        eval_results["map"],eval_result_weights["map"] = self._get_map()
        eval_results["mrr"],eval_result_weights["mrr"] = self._get_mrr()
        eval_results["f0.5"],eval_result_weights["f0.5"] = self._get_fmeasure(0.5)
        eval_results["f1"],eval_result_weights["f1"] = self._get_fmeasure(1)
        eval_results["f2"],eval_result_weights["f2"] = self._get_fmeasure(2)
        
        for metric in metrics:
            for n in tops_n:
                self.result_values[metric+"@"+str(n)] = eval_results[metric][n]
                self.result_weights[metric+"@"+str(n)] = eval_result_weights[metric][n]
    #
    #
    ###########################################################################
    # private methods
    #
    ###########################################################################
    # return [hr@5, hr@10, hr@15, hr@20, hr@30, hr@40, hr@50]
    def _get_hit_ratio(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        hri, wi = {}, {}
        for i in tops_n:
            hri[i], wi[i] = 0.0, nb_u
           
        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    if sum(self.rec_links_binary[u][:i]) >= 1:
                        hri[i] += 1
            for i in tops_n:
                hri[i] = (hri[i] * 1.0)/nb_u
                
        return hri, wi
    #
    ###########################################################################
    # return [precision@5, precision@10, precision@15, precision@20, precision@30, precision@40, precision@50]
    def _get_precision(self):
        preci, deno_preci, nume_preci, wi = {}, {}, {}, {}
        for i in tops_n:
            preci[i], deno_preci[i], nume_preci[i], wi[i] = 0.0, 0.0, 0.0, 0.0
            
        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # denominator [number of recommendations]
                deno_preci[i] += len(self.rec_links[u][:i])
                
                # numerator [number of good recommendations]
                nume_preci[i] += sum(self.rec_links_binary[u][:i])
        
        for i in tops_n:
            # compute the precision metric
            preci[i] = (1.0 * nume_preci[i])/(1.0 * deno_preci[i]) if deno_preci[i] > 0 else 0.0
            wi[i] = deno_preci[i]

        return preci, wi
    #
    ###########################################################################
    # return [recall@5, recall@10, recall@15, recall@20, recall@30, recall@40, recall@50]
    def _get_recall(self):
        recalli, deno_recalli, nume_recalli, wi = {}, {}, {}, {}
        for i in tops_n:
            recalli[i], deno_recalli[i], nume_recalli[i], wi[i] = 0.0, 0.0, 0.0, 0.0

        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # denominator [number of links observed]
                u_nb_links_to_rec = len(self.links_to_rec[u])
                deno_recalli[i] = deno_recalli[i] + u_nb_links_to_rec
                
                # numerator [number of good recommendations]
                nume_recalli[i] += sum(self.rec_links_binary[u][:i])
        
        for i in tops_n:
            # compute the recall metric
            recalli[i] = (1.0 * nume_recalli[i])/(1.0 * deno_recalli[i]) if deno_recalli[i] > 0 else 0.0
            wi[i] = deno_recalli[i]
        
        return recalli, wi
    #
    ###########################################################################
    # return [map@5, map@10, map@15, map@20, map@30, map@40, map@50]
    def _get_map(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        mapi, nume_mapi, wi, = {}, {}, {}
        for i in tops_n:
            mapi[i], nume_mapi[i], wi[i] = 0.0, 0.0, nb_u

        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    nume_mapi[i] += self._get_average_precision(self.rec_links_binary[u][:i])
            for i in tops_n:
                # compute the map metric
                mapi[i] = (1.0 * nume_mapi[i])/(1.0 * nb_u)
        return mapi, wi
    
    def _get_average_precision(self, user_rec_links_binary):
        average_precision = 0 # average precision
        indexes_good_rec = [index for index, val in enumerate(user_rec_links_binary) if val == 1]
        for index_val in indexes_good_rec:
            index_position = indexes_good_rec.index(index_val)
            average_precision += (1.0 * (index_position+1))/(1.0 * (index_val+1))
        average_precision = (1.0 * average_precision)/(1.0 * len(indexes_good_rec)) if len(indexes_good_rec) > 0 else 0.0
        return average_precision
    #
    ###########################################################################
    # Mean Reciprocal Rank
    # return [mrr@5, mrr@10, mrr@15, mrr@20, mrr@30, mrr@40, mrr@50]
    def _get_mrr(self):
        nb_u = 1.0 * len(self.rec_links_binary)
        mrri, nume_mrri, wi, = {}, {}, {}
        for i in tops_n:
            mrri[i], nume_mrri[i], wi[i] = 0.0, 0.0, nb_u
        
        if nb_u > 0:
            for u in self.rec_links_binary.keys():
                for i in tops_n:
                    nume_mrri[i] += self._get_reciprocal_rank(self.rec_links_binary[u][:i])
            for i in tops_n:
                # compute the mrr metric
                mrri[i] = nume_mrri[i]/(1.0 * nb_u)
        return mrri, wi
    
    def _get_reciprocal_rank(self, user_rec_links_binary):
        reciprocal_rank = 0.0 # reciprocal rank
        if 1 in user_rec_links_binary:
            index_first_good_rec = user_rec_links_binary.index(1)
            reciprocal_rank = 1.0/(1.0 * (index_first_good_rec+1))
        return reciprocal_rank
    #
    ###########################################################################
    # F-Measure
    # return [fm@5, fm@10, fm@15, fm@20, fm@30, fm@40, fm@50]
    def _get_fmeasure(self, b):
        
        true_positivei, false_negativei, false_positivei, fmi, wi = {}, {}, {}, {}, {}
        for i in tops_n:
            true_positivei[i], false_negativei[i], false_positivei[i], fmi[i], wi[i] = 0.0, 0.0, 0.0, 0.0, 0.0

        for u in self.rec_links_binary.keys():
            for i in tops_n:
                # True positive [number of good recommendations]
                true_positivei[i] += sum(self.rec_links_binary[u][:i])
                
                # False negative [number of links observed but which are not predicted]
                false_negativei[i] += (len(self.links_to_rec[u]) - true_positivei[i])
                
                # False positive [predicted True but which are False]
                false_positivei[i] += (i - true_positivei[i])
        
        for i in tops_n:   
            # numerator of F-measure
            fmi[i] = (1 + b*b) * true_positivei[i]
    
            # denominator of F-measure
            wi[i] = ((1 + b*b) * true_positivei[i]) + (b*b * false_negativei[i]) + false_positivei[i]
                    
            fmi[i] = (1.0 * fmi[i])/(1.0 * wi[i]) if wi[i] > 0 else 0.0
        
        return fmi, wi
#
###############################################################################
###############################################################################
# 
# PAGERANK
#
###############################################################################
###############################################################################
#
import networkx as nx
import scipy.sparse
#max_iter=100
def pagerank_scipy(G, alpha=0.85, personalization=None, max_iter=20, tol=1.0e-6, weight='weight', dangling=None):

    N = len(G)
    if N == 0:
        return {}

    nodelist = list(G)
    M = nx.to_scipy_sparse_matrix(G, nodelist=nodelist, weight=weight, dtype=float)
    S = scipy.array(M.sum(axis=1)).flatten()
    S[S != 0] = 1.0 / S[S != 0]
    Q = scipy.sparse.spdiags(S.T, 0, *M.shape, format='csr')
    M = Q * M

    # initial vector
    x = scipy.repeat(1.0 / N, N)

    # Personalization vector
    if personalization is None:
        p = scipy.repeat(1.0 / N, N)
    else:
        p = scipy.array([personalization.get(n, 0) for n in nodelist], dtype=float)
        p = p / p.sum()

    # Dangling nodes
    if dangling is None:
        dangling_weights = p
    else:
        # Convert the dangling dictionary into an array in nodelist order
        dangling_weights = scipy.array([dangling.get(n, 0) for n in nodelist], dtype=float)
        dangling_weights /= dangling_weights.sum()
    is_dangling = scipy.where(S == 0)[0]

    # power iteration: make up to max_iter iterations
    for _ in range(max_iter):
        xlast = x
        x = alpha * (x * M + sum(x[is_dangling]) * dangling_weights) + (1 - alpha) * p
        
        # check convergence, l1 norm
        err = scipy.absolute(x - xlast).sum()
        if err < N * tol:
            return dict(zip(nodelist, map(float, x)))
    return dict(zip(nodelist, map(float, x))) #raise nx.PowerIterationFailedConvergence(max_iter)
#
###############################################################################
###############################################################################
# 
# EXPGEN
#
###############################################################################
###############################################################################
#
from sklearn.metrics import jaccard_similarity_score

class Expgen (object):
    
    def __init__ (self, linkstream, slice_length, recsys_list, trust_network, rating_max):
        
        if len(linkstream) < 3: raise ValueError('Expgen : linkstream bad value!')
        if slice_length < 10: raise ValueError('Expgen : slice_length bad value!')
        if len(recsys_list) < 1: raise ValueError('Expgen : recsys_list bad value!')

        self.linkstream = linkstream
        self.slice_length = slice_length
        self.recsys_list = recsys_list 
        
        self.tmin = linkstream[0][0]
        self.tmax = linkstream[-1][0]
        
        self.trust_network = trust_network
        self.rating_max = rating_max
        self.rating_median = rating_max/2.0
        
        self.exp_results = {}
        for recsys in self.recsys_list:
            self.exp_results[recsys.name] = {}

        
    def save(self, filename, resdir=None):
        save_file = Out(filename, resdir, "w")
        save_file.write(str(self.exp_results))
        save_file.write(str(len(self.linkstream)))
        save_file.write(str(len(self.recsys_list)))
        save_file.write(str(self.tmin))
        save_file.write(str(self.tmax))
        save_file.close()
        
    def get_exp_results(self):
        return self.exp_results
        
    def run(self):
        exp_tmin, slice_iter, link_iter, nb_all_links = self.tmin, 1, 0, len(self.linkstream)
        user_list, item_list, last_user_item, last_item_list = set(), set(), {}, {}
        content_list = set()
        last_user_rating_mean, last_user_list_id = np.array([]), {}
        results_iter = {}

        # The initial time corresponds to the time of appearance of the first link of linkstream
        for recsys in self.recsys_list:
            recsys.tbegin = exp_tmin
            results_iter[recsys.name] = {}
            
        user_trust_global = {}
        for (u,trusted) in self.trust_network:
            if not u in user_trust_global.keys():
                user_trust_global[u] = []
            user_trust_global[u].append(trusted)
        cumulate_substream = []    
        while link_iter < nb_all_links:
            #
            # Extract substream and links to recommend
            #
            substream, links_to_rec = [], {}
            slice_tmax = exp_tmin + slice_iter * self.slice_length
            slice_new_user_list, slice_new_item_list, slice_new_user_item = set(), set(), {}
            slice_new_content_list = set()
            
            number_of_links_rec = 0
            while link_iter < nb_all_links:
                link = self.linkstream[link_iter]
                t, u, i, c, r = link[0], link[1], link[2], link[3], link[4]
                if t > slice_tmax:
                    break   
                substream.append(link)
                cumulate_substream.append(link)

                b_old_u, b_old_i = False, False
                if u in user_list:
                    b_old_u = True 
                else:
                    slice_new_user_list.add(u)
                    
                if i in item_list:
                    b_old_i = True 
                else:
                    slice_new_item_list.add(i)
                
                if not c in content_list:
                    slice_new_content_list.add(c)
                        
                if (b_old_u==True) and (b_old_i==True):
                    # old_user and old_item
                    if u in last_user_item.keys():
                        if not i in last_user_item[u]:
                            if r >= self.rating_median and r >= last_user_rating_mean[last_user_list_id[u]]:
                                if not u in links_to_rec.keys():
                                    links_to_rec[u] = set()
                                links_to_rec[u].add(i)
                                number_of_links_rec += 1
                        
                if not u in slice_new_user_item.keys():
                    slice_new_user_item[u] = set()
                slice_new_user_item[u].add(i)
        
                link_iter += 1
            
            ###################################################################
            # we have the substream of the the current time_slice
            print 'len(substream)', len(substream),'len(cumulate_substream)',len(cumulate_substream)
            user_list.update(slice_new_user_list)
            item_list.update(slice_new_item_list)
            content_list.update(slice_new_content_list)
            print slice_iter
            print 'nb_users_to_rec: ',len(links_to_rec.keys()),'number_of_links_rec: ',number_of_links_rec

            # creation of integer user_id and integer item_id for similarity_matrix
            user_list_id, id_user_list = {}, {}
            u_id = 0
            for u in user_list:
                user_list_id[u] = u_id
                id_user_list[u_id] = u
                u_id += 1
            item_list_id, id_item_list = {}, {}
            i_id = 0
            for i in item_list:
                item_list_id[i] = i_id
                id_item_list[i_id] = i
                i_id += 1
            content_list_id, id_content_list = {}, {}
            c_id = 0
            for c in content_list:
                content_list_id[c] = c_id
                id_content_list[c_id] = c
                c_id += 1
            
            # creation of information matrix
            nb_user, nb_item = len(user_list), len(item_list)
            rating_matrix = np.zeros(shape = (nb_user, nb_item))
            item_content = {}
            for (t, u, i, c, r) in cumulate_substream:
                rating_matrix[user_list_id[u], item_list_id[i]] = r 
                
                # for item contents
                if not i in item_content.keys():
                    item_content[i] = []
                item_content[i].append(c)
                
            # for user trust
            user_trust = {}
            for u in user_trust_global.keys():
                user_trust[u] = list(set(user_trust_global[u]) & user_list)#user_trust_global[u]
           
            # user information
            user_rating_mean = np.true_divide(rating_matrix.sum(1),(rating_matrix!=0.0).sum(1))
            implicit_rating_matrix = np.zeros(shape = (nb_user, nb_item))
            for u_i in range(nb_user):
                for i_i in range(nb_item):
                    rrr = rating_matrix[u_i, i_i]
                    if rrr >= self.rating_median and rrr >= user_rating_mean[u_i]:
                        implicit_rating_matrix[u_i,i_i] = 1
                    else:
                        implicit_rating_matrix[u_i,i_i] = 0
            
            user_jaccard_similarity = np.zeros(shape = (nb_user, nb_user))
            for u_i_1 in range(0,nb_user,1):
                for u_i_2 in range(u_i_1,nb_user,1):
                    sim_u1_u2 = jaccard_similarity_score(implicit_rating_matrix[u_i_1], implicit_rating_matrix[u_i_2])
                    user_jaccard_similarity[u_i_1,u_i_2] = sim_u1_u2
                    user_jaccard_similarity[u_i_2,u_i_1] = sim_u1_u2
            
            # global information on users and items
            global_info = {}
            #
            # users global information
            global_info['user_trust'] = user_trust#user_trust_global
            global_info['user_rating_mean'] = user_rating_mean
            global_info['user_similarity'] = user_jaccard_similarity
            #
            # rating info [max, median, min]
            global_info['rating_info'] = [self.rating_max, self.rating_max/2.0, 0.0]
            #
            # user_list_id , item_list_id and content_list_id
            global_info['user_list_id'] = user_list_id
            global_info['item_list_id'] = item_list_id
            global_info['id_user_list'] = id_user_list
            global_info['id_item_list'] = id_item_list
            #
            # number of distinct ratings 
            global_info['nb_ratings'] = np.count_nonzero(rating_matrix)
            global_info['rating_matrix'] = rating_matrix
            #
            ###################################################################
            #        
            # applied recommender systems for the current time slice           
            #
            for recsys in self.recsys_list:
                rec_links = recsys.get_recommended_list(links_to_rec.keys(), last_user_item, last_item_list)
                recsys_eval = Evaluation(links_to_rec, rec_links)
                recsys_eval.compute_evaluation_results()
                results_iter[recsys.name][slice_iter] = {"value":{} , "weight":{}}
                results_iter[recsys.name][slice_iter]["value"] = recsys_eval.get_result_values()
                results_iter[recsys.name][slice_iter]["weight"] = recsys_eval.get_result_weights()
                
                # update recsys for next step
                recsys.update_recsys(substream, cumulate_substream, global_info)
            #
            ###################################################################
            # 
            # preparation of the next time slice
            #
            last_user_rating_mean = np.copy(user_rating_mean)
            last_user_list_id = copy.deepcopy(user_list_id)
            last_item_list = copy.deepcopy(item_list)
            for u in slice_new_user_item.keys():
                if u in last_user_item.keys():
                    last_user_item[u].update(slice_new_user_item[u])
                else:
                    last_user_item[u] = slice_new_user_item[u]
            
            slice_iter += 1
            
            # supression of objects that were create during this iteration
            del substream[:]
            links_to_rec.clear()
            number_of_links_rec = 0
            slice_new_user_list.clear()
            slice_new_item_list.clear()
            slice_new_user_item.clear()
        #
        #######################################################################
        #
        #  compute time averaged values of any evaluation metric
        #
        eval_metrics = Evaluation.evaluation_metric_list()
        for recsys in self.recsys_list:
            for metric in eval_metrics:
                numerator,denominator = 0,0
                for slice_iter in results_iter[recsys.name].keys():
                    numerator += (results_iter[recsys.name][slice_iter]["value"][metric] * results_iter[recsys.name][slice_iter]["weight"][metric])
                    denominator += results_iter[recsys.name][slice_iter]["weight"][metric]
                self.exp_results[recsys.name][metric] = numerator/denominator if denominator > 0 else 0.0
        
        # supression of created dobjects
        user_list.clear()
        item_list.clear()
        last_user_item.clear()
        results_iter.clear()
#
###############################################################################
############################################################################### 
#
#  RECOMMENDER SYSTEMS BASIC GENERIC CLASS
#
###############################################################################
#
class Recsysgen (object):
    
    def __init__ (self, tbegin, recsys_id, name):
        self.tbegin = tbegin
        self.recsys_id = recsys_id
        self.name = name

    def update_recsys(self, substream, cumulate_substream, global_info):
        raise NotImplementedError("Subclasses should implement this!")
    
    def get_recommended_list(self, users_to_rec, user_item_list, all_items):
        raise NotImplementedError("Subclasses should implement this!")
#
###############################################################################
###############################################################################
# 
# GRAPHRS (RECSYS GRAPH)
#
###############################################################################
###############################################################################
#
#import sys
import os
import math
import ast 

###############################################################################
###############################################################################
#
# TIME-WEIGHT FUNCTIONS
#
def tfunction_identity(weight_init, Dt, nt, ta=None):
    return weight_init

def tfunction_half_life(weight_init, Dt, nt, ta=None):
    if nt > 0:
        return (weight_init * math.exp(-(math.log(2) * Dt * 1.0/nt)))
    return (weight_init * math.exp(-1 * Dt * math.log(2)))

def tfunction_logistic(weight_init, Dt, nt, ta=5):
    K = (ta*1.0)/nt if nt > 0 else ta
    return (1.0 - (1.0/ (1.0 + math.exp(-1.0 * K * (Dt - nt)))))

def tfunction_constant_decay(weight_init, Dt, nt, ta=None):
    tw = (1 - ((Dt * 1.0)/(nt * 1.0))) if (nt > 0 and Dt <= nt) else 0.0
    return tw

def tfunction_short_term(weight_init, Dt, nt, ta=None):
    tw = 1.0 if Dt < nt else 0.0
    return tw

time_weight_functions = [tfunction_identity, tfunction_half_life, tfunction_logistic, tfunction_constant_decay, tfunction_short_term]
#
###############################################################################
#
# GRAPH BASED RECOMMENDATIONS 
#
class GraphRecsys (Recsysgen):
    
    def __init__ (self, tbegin, recsys_id, name, graph_type=0, alpha=0, time=0, nt=0, ta=None,   
                        content=0, delta=0, beta=0, kp=0, k=0):
        
        super(GraphRecsys, self).__init__(tbegin, recsys_id, name)
        
        # additional attributes for all graph-based recsys
        #
        self.graph = nx.DiGraph()
        self.graph_type = graph_type # 0 for Bipartite, 1 for STG, 2 for LSG
        self.alpha = alpha # for the pagerank
        #
        # additional attributes for time weight
        #
        if time < 0: raise ValueError('_GraphRecsys : time bad value!')
        if nt < 0: raise ValueError('_GraphRecsys : nt bad value!')
        self.time = time #if len(time_weight_functions) > time else 0
        self.nt = nt if nt >= 1.0 else 1
        self.tfunction = time_weight_functions[time]
        self.ta = 0 if ta == None else ta
        #
        # additional parameters for content-based graphs
        #
        self.content = content
        #
        # additional parameters for session-based graphs
        #
        self.delta = delta if delta >= 1.0 else 1
        self.beta = beta
        self.user_last_sessions = {}
        #
        # additional parameters for continuous graph
        #
        self.item_last_sessions = {}
        self.content_last_sessions = {}
        #
        # additional parameters for User Common Behavior ( Interest )
        #
        self.k = k 
        self.kp = kp
        #
        # additional parameters for information linked only to user and information only linked to item
        #
        self.users_info = {}
        self.items_info = {}
        
        # users global information
        self.user_trust = ''
        self.user_rating_mean = ''
        self.user_pearson_similarity = ''
        #
        # rating info [max, median, min]
        self.rating_max, self.rating_median, self.rating_min = '', '', ''
        #
        # user_list_id item_list_id****
        self.user_list_id = ''
        self.item_list_id = ''
        self.id_user_list = ''
        self.id_item_list = ''
        
        self.rating_matrix = ''
    #
    ###########################################################################
    #
    def __str__(self):
        ucbtype = ""
        if self.k==0 : ucbtype="NA"
        elif self.k>0 : ucbtype="TopK"
        else: ucbtype="All"
        to_string = {
                "id":self.recsys_id, "name":self.name,
                "gtype":self.graph_type, "ctype":self.content, "twtype":self.time, "ucbtype":ucbtype,
                "delta": self.delta, "k": self.k, "beta": self.beta, "alpha": self.alpha, "nt": self.nt, "ta": self.ta
        }
        return str(to_string)
    #
    ###########################################################################
    #   
    def update_recsys(self, substream, cumulate_substream, global_info):

        # users global information
        self.user_trust = global_info['user_trust']
        self.user_rating_mean = global_info['user_rating_mean']
        self.user_pearson_similarity = global_info['user_similarity']
        #
        # rating info [max, median, min]
        self.rating_max, self.rating_median, self.rating_min = global_info['rating_info'][0], global_info['rating_info'][1], global_info['rating_info'][2]
        #
        # user_list_id item_list_id****
        self.user_list_id = global_info['user_list_id']
        self.item_list_id = global_info['item_list_id']
        self.id_user_list = global_info['id_user_list']
        self.id_item_list = global_info['id_item_list']
        #
        # rating matrix
        self.rating_matrix = global_info['rating_matrix']
        #   
        #######################################################################
        #
        # BIPARTITE GRAPH
        #
        #######################################################################
        #
        if self.graph_type == 0:
            #
            ###################################################################
            # NO CONTENT
            if self.content < 1:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0: 
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
            #
            ###################################################################
            # CONTENT WITH NCI ONLY    
            elif self.content == 1:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, cstr, weight=1, w_init=1, time=t)
            #
            ###################################################################
            # CONTENT WITH NCU ONLY    
            elif self.content == 2:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ustr, cstr, weight=1, w_init=1, time=t)
            #
            ###################################################################
            # CONTENT WITH NCI AND NCU    
            else:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ustr, cstr, weight=1, w_init=1, time=t)
        #   
        #######################################################################
        #
        # SESSION BASED TEMPORAL GRAPH
        #
        #######################################################################
        #
        elif self.graph_type == 1:
            if self.delta < 1: self.delta = 1
            #
            ###################################################################
            # NO CONTENT
            #print "STG Simple"
            if self.content < 1:
                #nn,ne = self.graph.number_of_nodes(),self.graph.number_of_edges()
                sstr_set,ustr_set,istr_set = set(),set(),set()
                T_set = set()
                #print "self.tbegin",self.tbegin,"self.delta",self.delta
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        T = 1 + int((t - self.tbegin)//self.delta)
                        T_set.add(T)
                        #print "--------->",T,t,self.tbegin,self.delta,int((t - self.tbegin)//self.delta)
                        s = (u,T)
                        ustr,istr,sstr = "u"+str(u),"i"+str(i),"s"+str(s)
                        sstr_set.add(sstr)
                        ustr_set.add(ustr)
                        istr_set.add(istr)
                        self.user_last_sessions[ustr] = sstr
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(sstr, node_type='s')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, sstr, weight=1, w_init=1, time=t)
                #nnp,nep = self.graph.number_of_nodes(),self.graph.number_of_edges()
                #print "size from",(nn,ne),"to",(nnp,nep)
                #print "sstr:",len(sstr_set),"ustr:",len(ustr_set),"istr:",len(istr_set),"T:",len(T_set)
                #print T_set
            #
            ###################################################################
            # CONTENT WITH NCI ONLY
            elif self.content == 1:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        T = int(1 + (t - self.tbegin)//self.delta)
                        sstr = "s"+str((str(link[1]),T)) # sstr = s(u,T)
                        self.user_last_sessions[ustr] = sstr
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_node(sstr, node_type='s')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, sstr, weight=1, w_init=1, time=t)
            #
            ###################################################################
            # CONTENT WITH NCU ONLY
            elif self.content == 2:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        T = int(1 + (t - self.tbegin)//self.delta)
                        sstr = "s"+str((str(link[1]),T)) # sstr = s(u,T)
                        self.user_last_sessions[ustr] = sstr
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_node(sstr, node_type='s')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ustr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, sstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, sstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, cstr, weight=1, w_init=1, time=t)
            #
            ###################################################################
            # CONTENT WITH NCI AND NCU
            else:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        T = int(1 + (t - self.tbegin)//self.delta)
                        sstr = "s"+str((str(link[1]),T)) # sstr = s(u,T)
                        self.user_last_sessions[ustr] = sstr
                        self.graph.add_node(ustr, node_type='u')
                        self.graph.add_node(istr, node_type='i')
                        self.graph.add_node(cstr, node_type='c')
                        self.graph.add_node(sstr, node_type='s')
                        self.graph.add_edge(ustr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, istr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(istr, sstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ustr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, ustr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(sstr, cstr, weight=1, w_init=1, time=t)
                        self.graph.add_edge(cstr, sstr, weight=1, w_init=1, time=t)
        #   
        #######################################################################
        #
        # LINKSTREAM GRAPH 
        #
        #######################################################################
        #
        elif self.graph_type == 2:
            #
            ###################################################################
            # NO CONTENT 
            if self.content < 1:  # continu sans content
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        ut,it = "u"+str((u,t)),"i"+str((i,t))
                        self.graph.add_node(ut, node_type='u')
                        self.graph.add_node(it, node_type='i')
                        self.graph.add_edge(ut, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ut, weight=1, w_init=1, time=t)
                        
                        if ustr in self.user_last_sessions.keys():
                            last_ut = self.user_last_sessions[ustr]
                            self.graph.add_edge(ut, last_ut, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ut, ut, weight=1, w_init=1, time=t)
                        
                        if istr in self.item_last_sessions.keys():
                            last_it = self.item_last_sessions[istr]
                            self.graph.add_edge(it, last_it, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_it, it, weight=1, w_init=1, time=t)
                        
                        self.user_last_sessions[ustr] = ut
                        self.item_last_sessions[istr] = it
            #
            ###################################################################
            # CONTENT WITH NCI ONLY
            elif self.content == 1:  
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        ut,it,ct = "u"+str((u,t)),"i"+str((i,t)),"c"+str((c,t))
                        self.graph.add_node(ut, node_type='u')
                        self.graph.add_node(it, node_type='i')
                        self.graph.add_node(ct, node_type='c')
                        self.graph.add_edge(ut, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ut, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ct, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ct, weight=1, w_init=1, time=t)

                        if ustr in self.user_last_sessions.keys():
                            last_ut = self.user_last_sessions[ustr]
                            self.graph.add_edge(ut, last_ut, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ut, ut, weight=1, w_init=1, time=t)
                        
                        if istr in self.item_last_sessions.keys():
                            last_it = self.item_last_sessions[istr]
                            self.graph.add_edge(it, last_it, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_it, it, weight=1, w_init=1, time=t)
                            
                        if cstr in self.content_last_sessions.keys():
                            last_ct = self.content_last_sessions[cstr]
                            self.graph.add_edge(ct, last_ct, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ct, ct, weight=1, w_init=1, time=t)
                        
                        self.user_last_sessions[ustr] = ut
                        self.item_last_sessions[istr] = it
                        self.content_last_sessions[cstr] = ct
            #
            ###################################################################
            # CONTENT WITH NCU ONLY
            elif self.content == 2:  
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        ut,it,ct = "u"+str((u,t)),"i"+str((i,t)),"c"+str((c,t))
                        self.graph.add_node(ut, node_type='u')
                        self.graph.add_node(it, node_type='i')
                        self.graph.add_node(ct, node_type='c')
                        self.graph.add_edge(ut, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ut, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ct, ut, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ut, ct, weight=1, w_init=1, time=t)
                        
                        if ustr in self.user_last_sessions.keys():
                            last_ut = self.user_last_sessions[ustr]
                            self.graph.add_edge(ut, last_ut, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ut, ut, weight=1, w_init=1, time=t)
                        
                        if istr in self.item_last_sessions.keys():
                            last_it = self.item_last_sessions[istr]
                            self.graph.add_edge(it, last_it, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_it, it, weight=1, w_init=1, time=t)
                            
                        if cstr in self.content_last_sessions.keys():
                            last_ct = self.content_last_sessions[cstr]
                            self.graph.add_edge(ct, last_ct, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ct, ct, weight=1, w_init=1, time=t)
                        
                        self.user_last_sessions[ustr] = ut
                        self.item_last_sessions[istr] = it
                        self.content_last_sessions[cstr] = ct
            #
            ###################################################################
            # CONTENT WITH NCI AND NCU
            else:
                for link in substream:
                    t,u,i,c,r = int(link[0]),str(link[1]),str(link[2]),str(link[3]),float(link[4])
                    ustr,istr,cstr = "u"+u,"i"+i,"c"+c
                    ui_w = rating_to_link_weight(u, r, self.user_rating_mean, self.user_list_id, self.rating_max)
                    if ui_w >= 0:
                        ut,it,ct = "u"+str((u,t)),"i"+str((i,t)),"c"+str((c,t))
                        self.graph.add_node(ut, node_type='u')
                        self.graph.add_node(it, node_type='i')
                        self.graph.add_node(ct, node_type='c')
                        self.graph.add_edge(ut, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ut, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ct, it, weight=1, w_init=1, time=t)
                        self.graph.add_edge(it, ct, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ut, ct, weight=1, w_init=1, time=t)
                        self.graph.add_edge(ct, ut, weight=1, w_init=1, time=t)
                        
                        if ustr in self.user_last_sessions.keys():
                            last_ut = self.user_last_sessions[ustr]
                            self.graph.add_edge(ut, last_ut, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ut, ut, weight=1, w_init=1, time=t)
                        
                        if istr in self.item_last_sessions.keys():
                            last_it = self.item_last_sessions[istr]
                            self.graph.add_edge(it, last_it, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_it, it, weight=1, w_init=1, time=t)
                            
                        if cstr in self.content_last_sessions.keys():
                            last_ct = self.content_last_sessions[cstr]
                            self.graph.add_edge(ct, last_ct, weight=1, w_init=1, time=t)
                            self.graph.add_edge(last_ct, ct, weight=1, w_init=1, time=t)
                        
                        self.user_last_sessions[ustr] = ut
                        self.item_last_sessions[istr] = it
                        self.content_last_sessions[cstr] = ct
        else:
            print "ERROR"
            pass
        #
        #######################################################################
        # TIME WEIGHT 
        tnow = int(substream[-1][0])
        self._time_weight(tnow)
   
###############################################################################    

    def get_recommended_list(self, users_to_rec, user_item_list, all_items):
        #print self.k
        rec = {}
        if len(users_to_rec) <= 0:
            return rec
        #   
        #######################################################################
        #
        # BIPARTITE GRAPH
        #
        #######################################################################
        #
        if self.graph_type == 0:
            if self.kp == 0: 
                all_items_istr = []
                for item in self.item_list_id.keys():
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    d[ustr] = 1 
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    d[ustr] = 0
                    
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
            elif self.kp == 1:
                all_items_istr = []
                for item in all_items:
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}

                for u in users_to_rec:
                    ustr = "u"+str(u)
                    # configuration of the personalized vector
                    d[ustr] = 1
                    u_trusted = []
                    if u in self.user_trust.keys():
                        u_trusted = self.user_trust[u]
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = self.k * 1
                        d[ustr] += (1 - self.k) * 1
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    
                    # reset the personalized vector to 0
                    d[ustr] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = 0
                    
                    # selection of new items
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
            else:
                all_items_istr = []
                for item in all_items:
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}

                for u in users_to_rec:
                    ustr = "u"+str(u)
                    # configuration of the personalized vector
                    d[ustr] = 1
                    u_trusted = user_item_list.keys()
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = self.k * 1 * self.user_pearson_similarity[self.user_list_id[u]][self.user_list_id[ui]]
                        d[ustr] += (1 - self.k) * 1
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    
                    # reset the personalized vector to 0
                    d[ustr] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = 0
                    
                    # selection of new items
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
                    
        #   
        #######################################################################
        #
        # SESSION BASED TEMPORAL GRAPH
        #
        #######################################################################
        #
        elif self.graph_type == 1:
            if self.kp == 0: 
                all_items_istr = []
                for item in all_items:
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    sstr = self.user_last_sessions[ustr]
                    d[ustr] = self.beta
                    d[sstr] = 1-self.beta 
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    d[ustr] = 0
                    d[sstr] = 0
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
            elif self.kp == 1: 
                all_items_istr = []
                for item in all_items:
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    sstr = self.user_last_sessions[ustr]
                    
                    # configuration of the personalized vector
                    d[ustr] = self.beta
                    d[sstr] = (1 - self.beta)
                    
                    u_trusted = []
                    if u in self.user_trust.keys():
                        u_trusted = self.user_trust[u]
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        ui_trust_weight = self.k * 1
                        d[uistr] = self.beta * ui_trust_weight * self.k
                        d[ustr] += self.beta * (1 - self.k)
                        if uistr in self.user_last_sessions.keys():
                            suistr = self.user_last_sessions[uistr]
                            d[suistr] = (1 - self.beta) * ui_trust_weight * self.k
                            d[sstr] += (1 - self.beta) * (1 - self.k)
                    
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    
                    # reset the personalized vector to 0
                    d[ustr] = 0
                    d[sstr] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = 0
                        if uistr in self.user_last_sessions.keys():
                            suistr = self.user_last_sessions[ustr]
                            d[suistr] = 0
                    
                    # selection of new items
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
            else: 
                all_items_istr = []
                for item in all_items:
                    all_items_istr.append("i"+str(item))
                    
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    sstr = self.user_last_sessions[ustr]
                    
                    # configuration of the personalized vector
                    d[ustr] = self.beta
                    d[sstr] = (1 - self.beta)
                    
                    u_trusted = user_item_list.keys()
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        ui_trust_weight = self.k * 1 * self.user_pearson_similarity[self.user_list_id[u]][self.user_list_id[ui]]
                        d[uistr] = self.beta * ui_trust_weight * self.k
                        d[ustr] += self.beta * (1 - self.k)
                        if uistr in self.user_last_sessions.keys():
                            suistr = self.user_last_sessions[uistr]
                            d[suistr] = (1 - self.beta) * ui_trust_weight * self.k
                            d[sstr] += (1 - self.beta) * (1 - self.k)
                    
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-15, max_iter=10)
                    
                    # reset the personalized vector to 0
                    d[ustr] = 0
                    d[sstr] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        d[uistr] = 0
                        if uistr in self.user_last_sessions.keys():
                            suistr = self.user_last_sessions[ustr]
                            d[suistr] = 0
                    
                    # selection of new items
                    new_items_indices = np.where(self.rating_matrix[self.user_list_id[u]] == 0.0)[0]
                    new_items = ['i'+str(self.id_item_list[item_id]) for item_id in new_items_indices]
                    new_items = list(set(new_items) & set(rank.keys()))
                    new_items_rank = {}
                    for item in new_items:
                        if rank[item] > 1e-15:
                            new_items_rank[item[1:]] = rank[item]
                    rec[u] = sorted(new_items_rank, key=new_items_rank.__getitem__, reverse=True)[:100]
                return rec
        #   
        #######################################################################
        #
        # LINKSTREAM GRAPH 
        #
        #######################################################################
        #
        elif self.graph_type == 2:
            if self.kp == 0:
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    last_ut = self.user_last_sessions[ustr]
                    d[last_ut] = 1 #
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-16, max_iter=30)
                    d[last_ut] = 0
                    
                    new_items = list(all_items - user_item_list[u])
                    
                    new_items_rank = {}
                    for node_time in rank.keys():
                        if node_time[:1]=="i":
                            item_time_tuple = ast.literal_eval(node_time[1:])
                            item = item_time_tuple[0]
                            if item in new_items:
                                if item in new_items_rank.keys():
                                    new_items_rank[item] += rank[node_time]
                                else:
                                    new_items_rank[item] = rank[node_time]
                    new_items_rank_final = {}
                    for item in new_items_rank.keys():        
                        if new_items_rank[item] > 1e-16:
                            new_items_rank_final[item] = new_items_rank[item]
                    rec[u] = sorted(new_items_rank_final, key=new_items_rank_final.__getitem__, reverse=True)[:100]    
                return rec
            elif self.kp == 1:
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    last_ut = self.user_last_sessions[ustr]

                    # configuration of the personalized vector
                    d[last_ut] = 1
                    
                    u_trusted = []
                    if u in self.user_trust.keys():
                        u_trusted = self.user_trust[u]
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        ui_trust_weight = self.k * 1
                        if uistr in self.user_last_sessions.keys():
                            last_uit = self.user_last_sessions[uistr]
                            d[last_uit] = ui_trust_weight * self.k
                            d[last_ut] += 1 * (1 - self.k)
                    
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-16, max_iter=30)
                    
                    # reset the personalized vector to 0
                    d[last_ut] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        if uistr in self.user_last_sessions.keys():
                            last_uit = self.user_last_sessions[uistr]
                            d[last_uit] = 0
                    
                    # selection of new items
                    new_items = list(all_items - user_item_list[u])
                    
                    new_items_rank = {}
                    for node_time in rank.keys():
                        if node_time[:1]=="i":
                            item_time_tuple = ast.literal_eval(node_time[1:])
                            item = item_time_tuple[0]
                            if item in new_items:
                                if item in new_items_rank.keys():
                                    new_items_rank[item] += rank[node_time]
                                else:
                                    new_items_rank[item] = rank[node_time]
                    new_items_rank_final = {}
                    for item in new_items_rank.keys():        
                        if new_items_rank[item] > 1e-16:
                            new_items_rank_final[item] = new_items_rank[item]
                    rec[u] = sorted(new_items_rank_final, key=new_items_rank_final.__getitem__, reverse=True)[:100]    
                return rec
            else:
                nodes = nx.nodes(self.graph)
                d = dict((node,0) for node in nodes)
                rec = {}
                for u in users_to_rec:
                    ustr = "u"+str(u)
                    last_ut = self.user_last_sessions[ustr]

                    # configuration of the personalized vector
                    d[last_ut] = 1
                    
                    u_trusted = user_item_list.keys()
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        ui_trust_weight = self.k * 1 * self.user_pearson_similarity[self.user_list_id[u]][self.user_list_id[ui]]
                        if uistr in self.user_last_sessions.keys():
                            last_uit = self.user_last_sessions[uistr]
                            d[last_uit] = ui_trust_weight * self.k
                            d[last_ut] += 1 * (1 - self.k)
                    
                    sum_d = sum(d.values()) * 1.0
                    for key in d.keys():
                        d[key] = (d[key] * 1.0)/sum_d
                    
                    # computation of pagerank
                    rank = pagerank_scipy(self.graph, alpha=self.alpha, personalization=d, tol=1e-16, max_iter=30)
                    
                    # reset the personalized vector to 0
                    d[last_ut] = 0
                    for ui in u_trusted:
                        uistr = "u"+str(ui)
                        if uistr in self.user_last_sessions.keys():
                            last_uit = self.user_last_sessions[uistr]
                            d[last_uit] = 0
                    
                    # selection of new items
                    new_items = list(all_items - user_item_list[u])
                    
                    new_items_rank = {}
                    for node_time in rank.keys():
                        if node_time[:1]=="i":
                            item_time_tuple = ast.literal_eval(node_time[1:])
                            item = item_time_tuple[0]
                            if item in new_items:
                                if item in new_items_rank.keys():
                                    new_items_rank[item] += rank[node_time]
                                else:
                                    new_items_rank[item] = rank[node_time]
                    new_items_rank_final = {}
                    for item in new_items_rank.keys():        
                        if new_items_rank[item] > 1e-16:
                            new_items_rank_final[item] = new_items_rank[item]
                    rec[u] = sorted(new_items_rank_final, key=new_items_rank_final.__getitem__, reverse=True)[:100]    
                return rec

        #######################################################################
    
    def _time_weight(self, tnow=None):
        if tnow!=None and self.time > 0:
            for (u,v) in self.graph.edges():
                Dt = tnow - self.graph[u][v]['time']
                weight_init = self.graph[u][v]['w_init']
                self.graph[u][v]['weight'] = self.tfunction(weight_init, Dt, self.nt, self.ta)


###############################################################################

#
###############################################################################
###############################################################################
#
def rating_to_link_weight(u, r, user_rating_mean, user_list_id, max_rating):
    r_u_mean = user_rating_mean[user_list_id[u]]
    num = r - r_u_mean
    deno = max_rating - r_u_mean 
    if deno > 0:
        return num/(1.0 * deno)
    if num >= 0:
        return 1.0
    return 0
#
###############################################################################
###############################################################################
# 
# LAUNCH RECSYS
#
###############################################################################
###############################################################################
###############################################################################
###############################################################################
#
#import sys
import os
#import ast
import copy
#import time


def get_recsys(recsys_id):

    graph = ['bip', 'stg', 'lsg']#, 'lsg2']
    content = ['na', 'ci', 'cu', 'ciu']
    time_weight = ['na', 'edf', 'lgf']#, 'cdf', 'stf']
    common_interest = ['na', 'tr', 'ts']#, 'ss']
    
    graph_param = {'bip':['alpha'], 'stg':['alpha','delta','beta'], 'lsg':['alpha']}#, 'lsg2':['alpha','ns']}
    content_param = {'na':[], 'ci':[], 'cu':[], 'ciu':[]}
    time_weight_param = {'na':[], 'edf':['nt'], 'lgf':['nt', 'ta']}#, 'cdf':['nt'], 'stf':['nt']}
    common_interest_param = {'na':[], 'tr':['k'], 'ts':['k']}#, 'ss':['k']}
    
    g_id, c_id, tw_id, ci_id = int(recsys_id[0]),int(recsys_id[1]),int(recsys_id[2]),int(recsys_id[3])
    
    recsys_name = graph[g_id]+"-"+content[c_id]+"-"+time_weight[tw_id]+"-"+common_interest[ci_id]
    
    recsys = GraphRecsys(0, recsys_id, recsys_name, graph_type=g_id, content=c_id, time=tw_id, kp=ci_id)
    
    param_list = []
    param_list.extend(graph_param[graph[g_id]])
    param_list.extend(content_param[content[c_id]])
    param_list.extend(time_weight_param[time_weight[tw_id]])
    param_list.extend(common_interest_param[common_interest[ci_id]])
    
    return recsys, param_list

###############################################################################

def main(rs_list, linkstream, trust_network, eval_metric, number_of_time_slices): 
    tmin = linkstream[0][0]
    tmax = linkstream[-1][0]
    if number_of_time_slices <= 2:
        number_of_time_slices = 2
    slice_length = int((tmax-tmin+number_of_time_slices)/number_of_time_slices)
     
    dict_graph = {"BIP":0, "STG":1, "LSG":2}
    dict_content = {"NA":0, "CI":1, "CIU":3}
    dict_time = {"NA":0, "EDF":1, "LDF":2}
    dict_trust = {"NA":0, "ET":1, "IT":2}

    recsys_list = []
    for rs_param in rs_list:
        rs_id = str(dict_graph[rs_param['graph']])+str(dict_content[rs_param['content']])
        rs_id += str(dict_time[rs_param['time']])+str(dict_trust[rs_param['trust']])
        rs_name = rs_param['graph']+"-"+rs_param['content']+"-"+rs_param['time']+"-"+rs_param['trust']
        recsys, param_list = get_recsys(rs_id)
        suffix = ''
        for param in param_list:
            if param=='delta':#Delta
                setattr(recsys, 'delta', rs_param['delta']*86400)
                suffix += "-Delta-"+str(rs_param['delta'])
            elif param=='beta':#Beta
                setattr(recsys, 'beta', rs_param['beta'])
                suffix += "-Beta-"+str(rs_param['beta'])
            elif param=='nt':#T0
                setattr(recsys, 'nt', rs_param['t0']*86400)
                suffix += "-T0-"+str(rs_param['t0'])
            elif param=='ta':#K
                setattr(recsys, 'ta', rs_param['k'])
                suffix += "-K-"+str(rs_param['k'])
            elif param=='k':#Gamma
                setattr(recsys, 'k', rs_param['gamma'])
                suffix += "-Gamma-"+str(rs_param['gamma'])
            elif param=='alpha':#Alpha
                setattr(recsys, 'alpha', rs_param['alpha'])
                suffix += "-Alpha-"+str(rs_param['alpha'])
    
        setattr(recsys, 'name', rs_name+suffix)
        recsys_list.append(recsys)
    
    timestamp = int((time.time())//1)
    resdir = "GraFC2T2"
    savefile = resdir+"-"+str(timestamp)

    print "BEGIN :",savefile

    param_exp = Expgen(linkstream, slice_length, recsys_list, trust_network, rating_max=5)
    param_exp.run()
    param_exp.save(savefile, resdir)
    
    file_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(file_dir,"out")
    repin = os.path.join(out,resdir)
    filename = os.path.join(repin, savefile+".txt")
    file_result = open(filename, 'r')
    recsys_results = ast.literal_eval(file_result.readline())
    
    recsys_results_file_save = open(resdir+"-RESULTS-"+str(timestamp)+".txt", "w")
    for rs in recsys_results.keys():
        line = "\n----> "+rs+"\n"
        recsys_results_file_save.write(line)
        for m in eval_metric:
            line = m+" = "+str(recsys_results[rs][m])+"\n"
            recsys_results_file_save.write(line)
    recsys_results_file_save.close()
    
    shutil.rmtree(out, ignore_errors=True)
    
    print "END :",savefile        
    

