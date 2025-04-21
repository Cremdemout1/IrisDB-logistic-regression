# logistic regression is the most widely used classification algorithm in industry
# because it works very well on linearly separable classes

#CLASSIFICATION MODEL

#LOG REG is readily generalized to multinomial logistic regression (softmax regression)
# more on this -> https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L05_gradient-descent_slides.pdf

#MATHEMATICS:

# odds = p / (1  p) where p stands for probability of positive event (event we want to predict)
# logit function = log-odds (logarithm of odds) = log

# logit function transforms probability between 0 and 1 to real numbers
# sigmoid function transforms real numbers into probabilities between 0 and 1

# sigmoid function -> σ(z) = 1 / 1 + e^-z where z is the net input(combination of weights)
# the activation function of linear regression is sigmoid function for binary classification
import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    return (1.0 / (1.0 + np.exp(-z)))

# z = np.arange(-7, 7, 0.1)
# phi_z = sigmoid(z)
# plt.plot(z, phi_z)
# plt.axvline(0.0, color='k')
# plt.ylim(-0.1, 1.1)
# plt.xlabel('z')
# plt.ylabel('$\phi (z)$')

# plt.yticks([0.0, 0.5, 1.0])
# ax = plt.gca()
# ax.yaxis.grid(True)
# plt.tight_layout()
# plt.show()

# log-likelihood function used to determine likelihood, which we want to maximize when building LOG REG
# log-likelihood: logL(w) = n Σ i=1 [y^i log(σ(z^i)) + (1 - y^i) log(1 - σ(z^i))] where σ(z^i) is sigmoid of each
# linear combination of weights and features

# It simplifies multiplication into addition (likelihood is a product of probabilities).
# It makes optimization easier (gradient descent works well on convex log-likelihood).
# Maximizing the log-likelihood is equivalent to minimizing the binary cross-entropy loss.

#from then on, we can either use gradient ascent to maximize log-likelihood
# OR 
# we can rewrite the log-likelihood function as a cost function that can be minimized using
# gradient descent like in previous models

# log-likelihood cost function (J):
# J(w) = n Σ i=1 [-y^i log(σ(z^i)) - (1 - y^i)log(1-σ(z^i))]

# def cost_1(z):
#     return - np.log(sigmoid(z))

# def cost_0(z):
#     return - np.log(1 - sigmoid(z))
# z = np.arange(-10, 10, 0.1)
# phi_z = sigmoid(z)
# c1 = [cost_1(x) for x in z]
# plt.plot(phi_z, c1, label='J(w) if y = 1')
# c0 = [cost_0(x) for x in z]
# plt.plot(phi_z, c0, linestyle='--', label='J(w) if y = 0')
# plt.ylim(0.0, 5.1)
# plt.xlim([0, 1])
# plt.xlabel('$\phi$(z)')
# plt.ylabel('J(w)')
# plt.legend(loc='best')
# plt.tight_layout()
# plt.show()

# above snippet demonstrates that the model penalizes wrong predictions with increasingly larger cost

# another way to write the negative log-likelihood (maximizing or minimizing its negative)
# is like the following: J(w) = -Σ i y^i log(σ(z^i)) + (1 - y^i ) log (1 - σ(z^i))

# Using logistic regression instead of an Adaptive Linear Neuron (Adaline):

class logisticregressionGD(object):
    """logistic regression classifier using gradient descent.
    
    Parameters:
    -----------
    eta: float
        learning rate between 0.0 and 1.0
    n_iter : int
        number of epochs (passes over training set)
    random_state : int
        random number gen seed for random and reproducible weight initialization

    Attributes:
    -----------
    w_ : 1d-array
        Weights after fitting
    cost_ : list
        logistic cost function value in each epoch
    """
    def __init__(self, eta=0.5, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """ Fit training data.

        Parameters:
        -----------
        X : {array-like}, shape = [n_examples, n_features]
            training vectors
        y : array-like, shape = [n_examples]
            target values

        Returns:
        -----------    
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01,
                              size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input) #activtion function of all examples in X
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors) # transposing vector to allow dot product
            self.w_[0] += self.eta * errors.sum()

            # next, we will compute the logistic 'cost' instead of the squared errors cost
            cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]
    
    def activation(self, z): #sigmoid function
        """compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
    
from sklearn import datasets
import numpy as np

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print ('Class labels: ', np.unique(y))

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, test_idx=None, resolutions=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'grey', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #plotting decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolutions), np.arange(x2_min, x2_max, resolutions))

    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=colors[idx], marker=markers[idx], label=cl, edgecolor='black')

    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], facecolors='none', edgecolor='black', alpha=1.0, linewidth=1, marker='o', s=100, label='test set')


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)
print("Labels counts in y: ", np.bincount(y))
print("Labels counts in y_train: ", np.bincount(y_train))
print("Labels counts in y_test: ", np.bincount(y_test))


X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = logisticregressionGD(eta=0.5, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel("petal length [standarized]")
plt.ylabel("petal width [standarized]")
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from sklearn.metrics import classification_report

X_test_01_subset = X_test[(y_test == 0) | (y_test == 1)]
y_test_01_subset = y_test[(y_test == 0) | (y_test == 1)]

y_pred = lrgd.predict(X_test_01_subset)

print(classification_report(y_test_01_subset, y_pred, target_names=["Setosa", "Versicolor"]))
