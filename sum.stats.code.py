import allel, scipy, keras, pickle
from keras import backend as K
import numpy as np
from siamese_nucvar import SiameseNetwork
from sklearn import manifold
import pandas as pd
#from matplotlib import pyplot as plt

cor = lambda a,b: np.corrcoef(a,b)[1][0]
spearman = lambda a,b: scipy.stats.spearmanr(a,b)[0]

def indv_ord_augmentation(almt):
            q = almt.copy()
            np.random.shuffle(q.T)
            return q

a = np.load('fixed.big.theta_sim.npz')
x_train, x_test = [a[i] for i in ['xtrain', 'xtest']]

#print(x_train[1].shape)
img_rows, img_cols = x_train[1].shape[0], x_train[1].shape[1]
print(img_cols, img_rows)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

def kelly_z(hap):
    def pairwise(li):  
        for i in range(len(li)):
            j = i+1
            while j < len(li):
                yield (li[i], li[j])
                j += 1
    r2 = lambda a,b: np.corrcoef(a,b)[1][0]**2
    l = list(range(hap.shape[0]))
    return np.mean([r2(hap[i,],hap[j,]) for i,j in pairwise(l)])
        
def adj_snp_r2(hap):
    r2 = lambda a,b: np.corrcoef(a,b)[1][0]**2
    def adj_pairs(li):
        i,j = 0,1
        while j < len(li):
            yield(i,j)
            i+=1
            j+=1
    l = list(range(hap.shape[0]))
    return np.mean([r2(hap[i,],hap[j,]) for i,j in adj_pairs(l)])

def uniq_snp_patterns(hap):
    s = set()
    for i in range(hap.shape[0]):
        snp_pattern = ' '.join(map(str,hap[i]))
        s.add(snp_pattern)
    return(len(s))

sum_stats_D = {"mean_exp_het":[],
               "uniq_haps":[],
               'mean_hap_ct':[],
               #'theta_pi':[],
               'seg_sites':[], 
               #'theta_W':[],
               #'mean_MAF':[],
               'tajD':[],
               'adj_LD':[],
               'mat_sum':[],
               'uniq_snp_pattern_ct':[]} 


for idx,mat in enumerate(x_test):
    mat = mat.reshape((mat.shape[0],mat.shape[1]))
    m = min(np.argwhere(mat.sum(axis=1)==0))[0]
    mat_all_seg = mat[:m,]
    #print(mat.shape, mat_all_seg.shape)
    #print(np.argwhere())
    hap = allel.HaplotypeArray(mat_all_seg)
    ac = hap.count_alleles()
    af = ac.to_frequencies()
    #print(hap.shape)
    
    mean_exp_het = allel.heterozygosity_expected(af, ploidy=2).mean()
    sum_stats_D["mean_exp_het"].append(mean_exp_het)
    
    uniq_haps = len(hap.distinct())
    sum_stats_D["uniq_haps"].append(uniq_haps)
    
    mean_hap_ct = np.mean([len(i) for i in hap.distinct()])
    sum_stats_D['mean_hap_ct'].append(mean_hap_ct)
    
 #   theta_pi = allel.mean_pairwise_difference(ac).mean()
 #   sum_stats_D['theta_pi'].append(theta_pi)
    
    poly_sites = mat_all_seg.shape[0]
    sum_stats_D['seg_sites'].append(poly_sites)
    
#    theta_W = poly_sites / sum([(i+1)**-1 for i in range(hap.shape[1])]) # this term is always 4.278543038936376 b/c each matrix has 40 indv
#    sum_stats_D['theta_W'].append(theta_W)
    
 #   mean_MAF = np.mean([min(i) for i in af])
 #   sum_stats_D['mean_MAF'].append(mean_MAF)
    
    tajD = allel.tajima_d(ac)
    sum_stats_D['tajD'].append(tajD)
    
    kz = kelly_z(hap)
    #print(kz)
    if 'kelly_z' not in sum_stats_D: sum_stats_D['kelly_z'] = []
    sum_stats_D['kelly_z'].append(kz)
    
    adj_LD = adj_snp_r2(hap)
    sum_stats_D['adj_LD'].append(adj_LD)
    
    sum_stats_D['mat_sum'].append(np.sum(hap))
    
    uniq_snp_pattern = uniq_snp_patterns(hap)
    sum_stats_D['uniq_snp_pattern_ct'].append(uniq_snp_pattern)
    print(idx)#, uniq_snp_patterns(hap.T), uniq_haps)


model = keras.models.load_model("3D.no.drop.out.fixed.big.8.epoch.test_base")
model.compile()

emb = model.predict(x_test)
#for pos in range(emb.shape[1]):
#    feat_vec = emb[:,pos]
#    out[pos] = list(feat_vec)

print(emb.shape)
# out = {"sum_stats":sum_stats_D}
# for i in sum_stats_D:
#     sum_stat_vec = sum_stats_D[i]
#     for pos in range(emb.shape[1]):
#         feat_vec = emb[:,pos]
#         print(i, pos, cor(sum_stat_vec, feat_vec), spearman(sum_stat_vec, feat_vec))
#         out[i+'|'+str(pos)] = {'pearson':cor(sum_stat_vec, feat_vec),
#                                'spearman':spearman(sum_stat_vec, feat_vec)}
#json.dump(out, open('sum.stats.features.and.all.pairwise.cor.json', 'w'), indent=1)
#pickle.dump(out, open('sum.stats.features.and.all.pairwise.cor.pickle', 'w'))
pd.DataFrame(emb).to_csv('3D.features.csv')
pd.DataFrame(sum_stats_D).to_csv('3D.sum.stats.csv')
