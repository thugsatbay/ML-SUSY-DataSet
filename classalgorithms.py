from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
from sklearn.naive_bayes import GaussianNB

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'notusecolumnones': False}
        self.reset(parameters)
        self.sigma=[]
        self.variance=[]
        self.probY=[]
        #self.skGNB=[]

    def reset(self, parameters):
        self.resetparams(parameters)
        # TODO: set up required variables for learning
        
    def learn(self,Xtrain,Ytrain):
        if self.params['notusecolumnones']:
            Xtrain=Xtrain[:,np.array(xrange(Xtrain.shape[1]-1))]
        y1Samples,y0Samples=float(len(Ytrain[Ytrain==1])),float(len(Ytrain[Ytrain==0]))
        probY1,probY0=y1Samples/len(Ytrain),y0Samples/len(Ytrain)
        sigma1=np.sum(Xtrain[Ytrain==1,],axis=0)/y1Samples
        sigma0=np.sum(Xtrain[Ytrain==0,],axis=0)/y0Samples
        variance1=np.sum(np.square(Xtrain[Ytrain==1,]-sigma1),axis=0)/y1Samples
        variance0=np.sum(np.square(Xtrain[Ytrain==0,]-sigma0),axis=0)/y0Samples
        

        #gnb = GaussianNB()
        #self.skGNB = gnb.fit(Xtrain, Ytrain)
        
        self.sigma=[sigma0,sigma1]
        self.variance=[variance0,variance1]
        self.probY=[probY0,probY1]


    def predict(self,Xtest):
        
        if self.params['notusecolumnones']:
            Xtest=Xtest[:,np.array(xrange(Xtest.shape[1]-1))]
        #ttemp=self.skGNB.predict(Xtest)
        '''
        print self.skGNB.theta_
        print self.sigma
        print ""
        print ""
        print self.skGNB.sigma_
        print self.variance
        '''
        dim=[Xtest.shape[0],Xtest.shape[1]]
        probNB1=[[utils.calculateprob(Xtest[r,c],self.sigma[1][c],math.sqrt(self.variance[1][c])) for c in xrange(dim[1])] for r in xrange(dim[0])]
        #print "Shape Of probNB1",len(probNB1),len(probNB1[0])
        probNB0=[[utils.calculateprob(Xtest[r,c],self.sigma[0][c],math.sqrt(self.variance[0][c])) for c in xrange(dim[1])] for r in xrange(dim[0])]
        NB1MLPR=(np.prod(probNB1,axis=1))*self.probY[1]
        #print "Shape of NB1MLPR",len(NB1MLPR)
        #print NB1MLPR
        NB0MLPR=(np.prod(probNB0,axis=1))*self.probY[0]
        ytest=[1 if NB1MLPR[x]>NB0MLPR[x] else 0 for x in xrange(dim[0])]
        return ytest

    # TODO: implement learn and predict functions                  
            
class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None','iterations':1000,'step-size':1,'tolerance':.000001}
        self.reset(parameters)

    def _costFunction(self,Xtrain,Ytrain,tempWeights):
        #print Xtrain.shape,tempWeights.shape
        #print np.dot(Xtrain,tempWeights).shape
        return np.sum(((-Ytrain)*np.log(utils.sigmoid(np.dot(Xtrain,tempWeights))))-((1-Ytrain)*(np.log(1-utils.sigmoid(np.dot(Xtrain,tempWeights))))))/Xtrain.shape[0]

    def _gradientDescentFunction(self,Xtrain,Ytrain,tempWeights):
        return np.dot(Xtrain.T,utils.sigmoid(np.dot(Xtrain,tempWeights))-Ytrain)/Xtrain.shape[0]

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))
     
    def learn(self,Xtrain,Ytrain):
        dim=(Xtrain.shape[0],Xtrain.shape[1]) #n*d
        Ytrain=Ytrain.reshape(dim[0],1) #n*1
        self.weights=np.array(np.random.random_sample((dim[1],))).reshape(dim[1],1) #d*1
        errorVal=self._costFunction(Xtrain,Ytrain,self.weights)
        for runs in xrange(self.params['iterations']):
            sstemp=self.params['step-size']
            newErrorVal=self._costFunction(Xtrain,Ytrain,self.weights)
            while newErrorVal>=errorVal:
                wtemp=self.weights-((sstemp)*self._gradientDescentFunction(Xtrain,Ytrain,tempWeights=self.weights))
                newErrorVal=self._costFunction(Xtrain,Ytrain,wtemp)
                sstemp=sstemp/2
            if runs%50==0:
                print "Logistic Regresssion Error Value",newErrorVal,"Step-Size",sstemp*2
            errorVal=newErrorVal
            self.weights=wtemp
            if errorVal<self.params['tolerance']:
                print "Tolerance Reached At Run",runs
                break

    def predict(self,Xtest):
        ytest = utils.sigmoid(np.dot(Xtest, self.weights))
        ytest[ytest >= .5] = 1     
        ytest[ytest < .5] = 0    
        return ytest
         
    # TODO: implement learn and predict functions                  
           

class NeuralNet(Classifier):

    #Stochastic Neural Net
    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None
        self.hiddenLayerHeight=None
        self.outputLayer=2
        self.epochs=None
        self.alpha=None
        if 'alpha' in self.params:
            self.alpha=self.params['alpha']
        else:
            self.alpha=0.1
        if 'epochs' in self.params:
            self.epochs=self.params['epochs']
        else:
            self.epochs=100
        if 'nh' in self.params:
            self.hiddenLayerHeight=self.params['nh']
        else:
            self.hiddenLayerHeight=4

    def learn(self,Xtrain,Ytrain):
        print "Neural Network Hidden Layer Height", self.hiddenLayerHeight
        dim=[Xtrain.shape[0],Xtrain.shape[1]]
        #initializing weights
        self.wi=np.array([np.random.random_sample((dim[1],)) for x in xrange(self.hiddenLayerHeight)]) #hL*d
        self.wo=np.array([np.random.random_sample((self.hiddenLayerHeight,)) for x in xrange(self.outputLayer)]) #k*hL
        print self.wi.shape
        print self.wo.shape
        reshapeSize=1
        Sigmoid=np.vectorize(lambda x: utils.sigmoid(x))
        for ep in xrange(self.epochs):
            #randomSample=np.random.permutation(samples)
            for n in xrange(Xtrain.shape[0]):
                
                yOutput=Ytrain[n]
                if yOutput==0:
                    yOutput=np.array([1,0]).reshape(self.outputLayer,reshapeSize)
                else:
                    yOutput=np.array([0,1]).reshape(self.outputLayer,reshapeSize)
                a1=Xtrain[n,:].reshape(dim[1],1) #d*1
                #print "a1",a1.shape
                z1=np.dot(self.wi,a1) #hL*1
                #print "z1",z1.shape
                a2=utils.sigmoid(z1) #hL*1
                #print "a2",a2.shape
                z2=np.dot(self.wo,a2) #k*1
                #print "z2",z2.shape
                a3f=utils.sigmoid(z2) #k*1
                #print "a3",a3f.shape
                delta2=(a3f-yOutput).reshape(self.outputLayer,reshapeSize)*utils.dsigmoid(z2).reshape(self.outputLayer,reshapeSize) #k*1
                #print "delta2",delta2.shape
                upDateWouter=np.dot(delta2,a2.T) #k*hL
                delta1=np.array(utils.dsigmoid(z1)).reshape(self.hiddenLayerHeight,reshapeSize)*np.dot(self.wo.T,delta2).reshape(self.hiddenLayerHeight,reshapeSize) #hL*1
                #print "delta1",delta1.shape
                upDateWinner=np.dot(delta1,a1.T) #hL*d
                tAlpha=self.alpha
                self.wi=self.wi-(tAlpha*upDateWinner)
                self.wo=self.wo-(tAlpha*upDateWouter)      
            if ep%20==0:
                ytest=self.predict(Xtrain)
                correct = 0
                for i in range(len(ytest)):
                    if ytest[i] == Ytrain[i]:
                        correct += 1
                print "On Epoch",(ep+1)
                print "Accuracy For Training Model",(correct/float(len(ytest))) * 100.0


    def predict(self, Xtest):
        diffSigmoid=np.vectorize(lambda x: utils.dsigmoid(x))
        z1=np.dot(self.wi,Xtest.T) #hL*n
        a2=np.array([utils.sigmoid(z1[:,x]) for x in xrange(Xtest.shape[0])]) #n*hL
        z2=np.dot(self.wo,a2.T) #k*n
        a3f=np.array([utils.sigmoid(z2[:,x]) for x in xrange(Xtest.shape[0])]) #n*k
        ytest=np.array([1 if a3f[x,1]>a3f[x,0] else 0 for x in xrange(Xtest.shape[0])])
        return ytest
    # TODO: implement learn and predict functions                  

    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        
    # TODO: implement learn and predict functions                  
           
    
