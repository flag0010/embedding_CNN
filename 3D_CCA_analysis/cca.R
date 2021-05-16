require(CCA)
require(corrplot)
#setwd('/home/flagel/Desktop/siamese')
stats = read.csv('3D.sum.stats.csv')
features = read.csv('3D.features.csv')
stats = stats[,-1]
features = features[,-1]

cca = cc(features, stats) #from CCA pkg
comput(features, stats, cca)[3:6] #correlations between 

cca2 = cancor(features, stats) #R builtin func

ccx = as.matrix(features) %*% cca2$xcoef #these are the 3 X canonical variates
ccy = as.matrix(stats) %*% cca2$ycoef # and these are the 3 Y canonical variates
cca2$cor #correlations between X and Y canonical variates

t(cor(ccx, features)) #correlation between X variables (features), and X canonical variates
#same as below using CCA pkg, at least subject to axis rotation (e.g. -0.19, vs 0.19, as in PCA)
#also comput func. gives all pairwise X canon var vs X var, Y vs X, and Y vs Y, and X vs Y


#some plots of canonical variates vs variables
pairs(data.frame(stats, features))
pairs(data.frame(ccx, stats))
pairs(data.frame(ccy, features))
corrplot(cor(data.frame(features, stats)))
