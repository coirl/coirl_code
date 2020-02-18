#######################################################################################################################
#   file name: Ellipsoid
#
#   description:
#   this file defines the Ellipsoid class and methods.
#######################################################################################################################
# imports:
#######################################################################################################################
from qinfer.utils import mvee, ellipsoid_volume, in_ellipsoid
import numpy as np
#######################################################################################################################

# Ellipsoid class for MVEE CIRL algorithm
# Describes ellipsoid (Q,c), i.e. {x: (x-c)@Q^-1@(x-c)<=1}
# Init: input: dimension. creates Q,c for MVEE of ball around 0 with radius 1 w.r.t infinity norm
# update: input: a. updates Q,c to MVEE of intersection of current ellipsoid and {x: (x-c)@a>=0}
# volume: returns volume of ellipsoid
# inside: returns True/False if given point x is inside/outside of the ellipsoid NOTE: float error is real
# getQ,getc: returns copy of ellipsoid's cQ,c
##################################################################
class Ellipsoid:
    ##################################################################
    def __edges_of_infty_norm_rad_1_ball(self, dim):
        points = []
        for i in range(2**dim):
            b = bin(i)[2:].zfill(dim)
            mask = np.asarray([int(char) for char in b])
            pt = np.ones(dim) - 2*mask
            points.append(pt)
        return np.asarray(points)

    ##################################################################
    def __init__(self, dim):
        self.__dim = dim
#         self.__Q, self.__c = mvee(self.__edges_of_infty_norm_rad_1_ball(self.__dim))
        self.__Q = (1/self.__dim)*np.eye(self.__dim)
        self.__c = np.zeros(self.__dim)
        self.__Q = np.linalg.inv(self.__Q)

    ##################################################################
    def update(self, a):
        at = np.expand_dims(-1/(np.sqrt(a.T @ self.__Q @ a + 1e-10)) * a,1)
        self.__c = (self.__c - (1/(self.__dim+1) * self.__Q @ at).T)[0]
        self.__Q = (self.__dim**2/(self.__dim**2-1)) * (self.__Q - (2/(self.__dim+1)) * self.__Q@at@at.T@self.__Q)

    ##################################################################
    def volume(self):
        return ellipsoid_volume(invA = self.__Q)

    ##################################################################
    def inside(self, x):
        return in_ellipsoid(x,self.__Q,self.__c)

    ##################################################################
    def getQ(self):
        return self.__Q.copy()

    ##################################################################
    def getc(self):
        return self.__c.copy()
    ##################################################################
