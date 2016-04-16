import numpy as np
from scipy.spatial import distance
from scipy import stats
import random
import timeit
import itertools
from scipy import misc

import math

def nCr(n,r):
    f = math.factorial
    return f(n) / f(r) / f(n-r)


class LogReg():
    def _sigmoid(self,theta,X):
        return np.transpose(1.0/(1.0+np.exp(-np.dot(X,theta))))
    def getstuff(self):
        return self.theta_multi
    def _p(self,theta,data):
        p=self._sigmoid(theta,data)
        return 1*(p>0.5)
    def fit(self,data,labels,multiclass="multi",learn_rate=0.1,reg_pen=0.0001,max_iter=100,tol=1e-4,adj_learn_rate=True):
        self.learn_rate=learn_rate
        self.reg_pen=reg_pen
        self.max_iter=max_iter
        self.tol=tol
        self.multiclass=multiclass
        self.adj_learn_rate=adj_learn_rate
        classes=np.unique(labels)
        self.classes=classes
        n=data.shape[0]
        if len(classes)==2:
            self._BinaryLogistic(data,labels)
        elif multiclass=="multi":
            self._MultiLogistic(data,labels)
        elif multiclass=="ovr": # one versus rest
            ovr_theta=np.zeros((data.shape[1]+1,len(classes)))
            for i in range(len(classes)):
                samples=np.sum(labels==classes[i])
                digit=classes[i]
                print("Running class: " + str(digit))
                masked_labels=np.hstack((-1*np.ones(samples),(labels[labels!=digit])[:samples]))
                masked_labels[masked_labels!=-1]=0
                masked_labels[masked_labels==-1]=1
                sub_data=np.vstack((data[labels==digit],(data[labels!=digit])[:samples]))
                self._BinaryLogistic(sub_data,masked_labels)
                ovr_theta[:,i]=self.theta[:,0]
            self.ovr_theta=ovr_theta

        elif multiclass=="ovo": # one versus one
            ovo_theta=np.zeros((data.shape[1]+1,nCr(len(classes),2)))
            i=0
            for c in itertools.combinations(classes,2):
                sub_data=data[np.logical_or(labels==classes[c[0]],labels==classes[c[1]])]
                sub_labels=labels[np.logical_or(labels==classes[c[0]],labels==classes[c[1]])]
                sub_labels2=np.zeros(len(sub_labels))
                sub_labels2[sub_labels==classes[c[0]]]=0
                sub_labels2[sub_labels==classes[c[1]]]=1
                print(c)
                self._BinaryLogistic(sub_data,sub_labels2)
                ovo_theta[:,i]=self.theta[:,0]
                i+=1
            self.ovo_theta=ovo_theta
        else:
            print("error!")

    def _MultiLogistic(self,data,labels):
        learn_rate=self.learn_rate
        reg_pen=self.reg_pen
        max_iter=self.max_iter
        tol=self.tol
        X = np.hstack((np.ones((data.shape[0], 1)),data))
        theta=np.zeros((X.shape[1],len(self.classes)))
        gradient=np.zeros((X.shape[1],len(self.classes)))
        last_error=1
        precision=1
        learn_rate/=data.shape[1]
        iterations=0

        while np.abs(precision) > tol: #iterate until the cost stops changing
            for i in range(len(self.classes)):
                #gradient[:,i]=np.transpose(np.dot(np.ones(np.sum([labels==self.classes[i]]))-self._sigmoid(theta[:,i],X[labels==self.classes[i]]),X[labels==self.classes[i]]))
                gradient[:,i]=np.transpose(np.dot(1*(labels==self.classes[i])-self._sigmoid(theta[:,i],X),X))
                #print(gradient)
            theta_new=theta+learn_rate*(reg_pen*(-theta)+gradient)
            theta=theta_new
            iterations+=1
            error=np.max(np.abs(gradient))
            precision = last_error-error
            if self.adj_learn_rate==True:
                if precision>0: #increase the learning rate if we got closer to
                    learn_rate*=1.01
                else:
                    learn_rate/=2
            #if iterations%10==0:
            print("Iteration: " +str(iterations) + " Precision: " +str(precision))
            if iterations==max_iter:
                break
            last_error=error
        self.theta_multi=theta

    def _BinaryLogistic(self,data,labels):
        learn_rate=self.learn_rate
        reg_pen=self.reg_pen
        max_iter=self.max_iter
        tol=self.tol
        X = np.hstack((np.ones((data.shape[0], 1)),data))
        theta=np.zeros((X.shape[1],1))
        last_error=1
        precision=1
        learn_rate/=data.shape[1]
        iterations=0
        while np.abs(precision) > tol: #iterate until the cost stops changing
            gradient=np.transpose(np.dot(labels-self._sigmoid(theta,X),X))
            theta_new=theta+learn_rate*(reg_pen*(-theta)+gradient)
            theta=theta_new
            iterations+=1
            error=np.max(np.abs(gradient))
            precision = last_error-error
            if self.adj_learn_rate==True:
                if precision>0: #increase the learning rate if we got closer to
                    learn_rate*=1.01
                else:
                    learn_rate/=2
            if iterations%10==0:
                print("Iteration: " +str(iterations) + " Precision: " +str(precision))
            if iterations==max_iter:
                break
            last_error=error
        self.theta=theta
    def predict(self,data):
        if len(self.classes)==2:
            return 1*(self.predict_proba(data)>0.5)
        elif self.multiclass=="multi":
            return np.argmax(self.predict_proba(data),axis=1)
        else:
            return np.argmax(self.predict_proba(data),axis=1)
    def predict_proba(self,data):
        X = np.hstack((np.ones((data.shape[0], 1)),data))
        num_classes=len(self.classes)
        if num_classes==2:
            return np.transpose(self._sigmoid(self.theta,X))
        if self.multiclass=="ovr":
            ovr_prob=np.zeros((data.shape[0],num_classes))
            for i in range(num_classes):
                ovr_prob[:,i]=np.transpose(self._sigmoid(self.ovr_theta[:,i],X))
            return ovr_prob
        elif self.multiclass=="ovo":
            ovo_prob=np.zeros((data.shape[0],num_classes*num_classes))
            i=0
            for c in itertools.combinations(self.classes,2):
                ovo_prob[:,num_classes*c[1]+c[0]]=np.transpose(self._sigmoid(self.ovo_theta[:,i],X))
                ovo_prob[:,num_classes*c[0]+c[1]]=1-ovo_prob[:,len(self.classes)*c[1]+c[0]]
                i+=1
            ovo_classprob=np.zeros((data.shape[0],num_classes))
            for i in range(num_classes):
                ovo_classprob[:,i]=np.sum(ovo_prob[:,i*num_classes:(i+1)*num_classes],axis=1)
            return ovo_classprob
        elif self.multiclass=="multi":
            return np.transpose(self._sigmoid(self.theta_multi,X))
    def score(self,data,labels):
        return np.sum(np.transpose(self.predict(data))==labels)/float(len(labels))


class TwoD_LDA():
    def fit(self,data,labels,r,c,rprime,cprime):
        self.r=r
        self.c=c
        self.rprime=rprime
        self.cprime=cprime

        uniq=np.unique(labels)
        pi=np.zeros(len(uniq))
        for i in range(len(uniq)):
            pi[i]=np.sum(data==uniq[i])
        features = data.shape[1]
        classes=len(uniq)
        n=data.shape[0]
        Mitemp=np.zeros((features,classes))
        for i in range(classes):
            cimages=data[labels==uniq[i]]
            Mitemp[:,i]= np.mean(cimages,axis=0)
        M=np.sum(pi*Mitemp,axis=1)


        M=M.reshape((r,c))/n
        Mi=np.zeros((classes,r,c))
        for i in range(classes):
            Mi[i,:,:]=Mitemp[:,i].reshape((r,c))


        X=np.zeros((n,r,c))
        for i in range(n):
            X[i,:,:]=data[i].reshape((r,c))

        print("s")

        R=np.random.rand(c,cprime)
        R_old=np.zeros((c,cprime))
        L_old=np.zeros((r,rprime))

        error=1
        while error >0.0001:
            S_w=np.zeros((r,c))
            S_b=np.zeros((r,c))
            RRt=np.dot(R,np.transpose(R))

            for cl in range(classes):
                index = np.where(labels==uniq[cl])

                for i in np.nditer(index):
                    X_tilde = X[i,:,:]-Mi[cl,:,:]
                    S_w += np.dot(np.dot(X_tilde,RRt),np.transpose(X_tilde))

                M_tilde = Mi[cl,:,:]-M
                S_b += pi[cl]*np.dot(np.dot(M_tilde,RRt),np.transpose(M_tilde))

            l,V=np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))
            idx = l.argsort()[::-1]
            V = V[:,idx]

            L=V[:,:rprime]

            S_w=np.zeros((r,c))
            S_b=np.zeros((r,c))
            LLt=np.dot(L,np.transpose(L))

            for cl in range(classes):
                index = np.where(labels==uniq[cl])

                for i in np.nditer(index):
                    X_tilde = X[i,:,:]-Mi[cl,:,:]
                    S_w += np.dot(np.dot(np.transpose(X_tilde),LLt),X_tilde)

                M_tilde = Mi[cl,:,:]-M
                S_b += pi[cl]*np.dot(np.dot(np.transpose(M_tilde),LLt),M_tilde)

            l,V=np.linalg.eig(np.dot(np.linalg.inv(S_w),S_b))
            idx = l.argsort()[::-1]
            V = V[:,idx]

            R=V[:,:cprime]

            error = np.sum(np.abs(R-R_old))+np.sum(np.abs(L-L_old))
            print(error)
            R_old=R
            L_old=L

        self.R=R
        self.L=L


    def transform(self,X,**kwargs):

        n=X.shape[0]
        Y=np.zeros((n,self.rprime*self.cprime))
        X=np.zeros((n,self.r,self.c))
        for i in range(n):
            X[i,:,:]=X[i].reshape((self.r,self.c))

        for i in range(n):
            Y[i,:]=np.dot(np.dot(np.transpose(self.L),X[i,:,:]),self.R).reshape(self.rprime*self.cprime)

        return Y


class PCA():

    def fit(self,data):
        self.data=data
        self.dmean=np.mean(data,axis=0)
        X_tilde=data-self.dmean
        self.U,self.S,self.V=np.linalg.svd(X_tilde,full_matrices=False)
    def transform(self,X,**kwargs):
        S=np.diag(self.S)
        cum_s = np.cumsum(S*S)
        cum_s = cum_s/cum_s[-1]
        if "percent" in kwargs:
            s = np.argmax(cum_s>kwargs["percent"])+1
        elif "components" in kwargs:
            s = kwargs["components"]
        print("Matrix has been reduced to "+str(s)+ " dimensions")
        return np.dot(X-self.dmean,np.transpose(self.V[:s,:]))


class FDA():

    def fit(self,X,trainlabels):
        d = X.shape[1]
        Sw = np.zeros((d,d));
        Sb = np.zeros((d,d));
        n=np.zeros(10)
        mu=np.mean(X,axis=0)
        self.uniq=np.unique(trainlabels)
        for i in range(len(self.uniq)):
            Xi = X[trainlabels==self.uniq[i],:]
            n[i]=Xi.shape[0]
            mui=np.mean(Xi,axis=0)
            Xi_tilde = Xi - mui
            mui_tilde= mui-mu
            Sw = Sw + np.dot(np.transpose(Xi_tilde),Xi_tilde)
            Sb = Sb + n[i]*np.outer(mui_tilde,mui_tilde)
        l, v = np.linalg.eig(np.linalg.solve(Sw,Sb))
        idx = l.argsort()[::-1]
        self.l = l[idx]
        self.v = v[:,idx]
    def transform(self,X):
        return np.dot(X,self.v[:,:len(self.uniq)-1])

### define the linear weighting custom function for sklearn classifiers
def linear_weights(x):
        k=x.shape[1]
        x=x.transpose()
        weights=((x[k-1,:]-x)/(x[k-1,:]-x[0,:])).transpose()
        return weights


class Fold(object): # My own "Fold" object. This object contains the indices for training and validation
    def __init__(self):
        self.train = None
        self.val = None

    def train(self):
        return self.train

    def val(self):
        return self.val


# k-nearest neighbor function

class KNN:
    def fit(self,k,train,train_lab, test,t_lab=None, d_meas="euclidean",weights="equal"):
        """
    Parameters
    ----------
    k : largest k to be computed
    train : numpy array containing the training data
    train_lab : list or numpy array containing the labels for the training data
    test : numpy array containing the test data
    d_meas : default distance measure is euclidean, also available are 'cityblock', 'cosine', etc. see the
    numpy 'cdist' documentation for more options
    weights : "equal" for equal weighting, "linear" for linearly decreasing weights
    t_lab : OPTIONAL PARAMETER useful for debugging
    """
        test_size = len(test) # number of test observations
        test_lab = np.empty([k,test_size],dtype=int)
        begin = timeit.default_timer()
        uclasses=np.unique(train_lab)
        for i in range(test_size): #loop through all test observations
            distances = distance.cdist(train,[test[i]],d_meas) # calculate distances from test sample to all training data
            classes = np.argpartition(distances,tuple(range(k)),axis=None)
            for l in range(k): #calculate class predictions for 1 to k neighbors
                if weights=="linear": #linear weighting
                    d_n1=distances[classes[l+1]][0] #k+1 nearest neighbor
                    d_1=distances[classes[0]][0] # nearest neighbor
                    wdist=(d_n1-distances[classes[:l+1]])/(d_n1-d_1) #compute weights
                    neighbors=train_lab[classes[:l]] #get labels of nearest k neighbors
                    weightsums=np.zeros([len(uclasses)],dtype=float) #initialization
                    for a in np.unique(neighbors): #loop through the nearest k neighbors
                        weightsums[a]=sum(wdist[neighbors==a])  #sum up weights for each class
                    test_lab[l, i]=np.argmax(weightsums) #set test label to the class with the highest weight
                elif weights=="equal": #majority voting
                    test_lab[l, i] = stats.mode(train_lab[classes[:l+1]])[0]
            if i%10==0:
                stop = timeit.default_timer()
                print "index: " + str(i) + "/"+ str(test_size) +"  percent complete: "+ str((i*100)/test_size) \
                      +"%  estimated time remaining: " + str((stop-begin)*(test_size-i)/(i+1)) + " seconds"
        print "total time: " + str(stop-begin) + " seconds"
        self.test_lab=test_lab
        self.uniq=uclasses
    def predict(self,k):
        if k>len(self.uniq):
            print("You must choose a k value less than or equal to "+str(len(self.uniq))+" (the k value you provided during the fit)")
        else:
            print("Predictions using the "+str(k)+ " nearest neighbors")
            return self.test_lab[k-1,:]








def localkmeans(k,train,train_lab, test,d_meas="euclidean"):
    """
    k : largest k to be computed
    train : numpy array containing the training data
    train_lab : list or numpy array containing the labels for the training data
    test : numpy array containing the test data
    d_meas : default distance measure is euclidean, also available are 'cityblock', 'cosine', etc.
    see the numpy 'cdist' documentation for more options
    :return:
    """
    test_size = len(test)
    test_lab = np.empty([k,test_size])#,dtype=int)
    classes=np.unique(train_lab)
    localknn_images = np.zeros((k*len(classes),train.shape[1]),dtype=float)
    begin = timeit.default_timer()

    for i in range(test_size):

        distances = distance.cdist(train,[test[i]],d_meas)
        for j in classes:

            class_index=np.where(train_lab==j)
            class_distances=distances[class_index]
            class_images=train[class_index]
            sorted_indices = np.argpartition(class_distances,tuple(range(k)),axis=None)[:k]
            localknn_images[range(j,k*len(classes),len(classes))]=class_images[sorted_indices]

        iter_means= np.zeros((len(classes),train.shape[1]),dtype=float)

        for l in range(k):
            iter_means+=localknn_images[l*len(classes):(l+1)*len(classes)]
            kmeans_distances = distance.cdist(iter_means,[test[i]*(l+1)],d_meas)
            test_lab[l,i] = np.argmin(kmeans_distances)

        if i%10==0:
            stop = timeit.default_timer()
            print "index: " + str(i) + "/"+ str(test_size) +"  percent complete: "+ str((i*100)/test_size) \
                  +"%  estimated time remaining: " + str((stop-begin)*(test_size-i)/(i+1)) + " seconds"
    print "total time: " + str(stop-begin) + " seconds"
    return test_lab


def find_class_means(train,train_lab,d_meas="euclidean"):
    classes=np.unique(train_lab)
    print(classes)
    class_means = np.empty([len(classes),train.shape[1]],dtype=float)
    for i in range(len(classes)):
        class_means[i, :] = np.mean(train[train_lab==classes[i]],axis=0)
    return class_means



def compute_error(true_labels, pred_labels):
    return 1-sum(true_labels == pred_labels)/float(len(true_labels))


def kfolds(k, train, randomize=False):
    n = len(train)
    folds = [Fold() for _ in range(n)]

    if randomize:
        indices = random.shuffle(range(n))
    else:
        indices = range(n)

    for i in range(k):

        folds[i].val = np.array(indices[int(i*np.floor(n/k)):int((i+1)*np.floor(n/k))+1]).astype(int)
        folds[i].train = np.hstack((indices[:int(i*np.floor(n/k))],indices[int((i+1)*np.floor(n/k))+1:])).astype(int)

    return folds



