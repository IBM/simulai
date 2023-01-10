# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

import numpy as np
import scipy as sp
import scipy.spatial as spatial
import sympy as sy
import numexpr as ne

class Kansas:

    def __init__(self,points,centers, sigma2 ="auto",Kernel="gaussian", eps = 1e-8):

        """
        Initialize the radial basis function interpolator.

        Parameters
        ----------
        points : np.ndarray
            The points at which to interpolate.
        centers : np.ndarray
            The centers of the kernel functions.
        sigma2 : Union[str, float], optional
            The variance of the kernel functions. If set to "auto", the variance is calculated from the maximum distance
            between centers.
        Kernel : str, optional
            The type of kernel function to use.
        eps : float, optional
            The tolerance for eliminating elements from the interpolation matrices.

        """
    
        self.Nk = centers.shape[0] # number of centers of kernels
        self.Kernel= Kernel #kernel function or string specifiying the type of kernel function
        self.eps = eps #tolerance to take off interpolation matrices elements
        self.points = np.float32(points) # interpolation points
        self.centers = np.float32(centers) # kernel centers
        self.ndim = centers.shape[1] # dimension of points

        if self.ndim != points.shape[1] :
            print("mismatch dimension of kernel centers and points")

        self.f1_was_gen = False

        #Calculate the radial function for the combination of centers and interpolation points
        self.r2 = np.float32(spatial.distance.cdist(points,centers, 'sqeuclidean'))

        #calculate sigma based on the mean maximal distance beteewn 2 kernels or use the informed sigma
        if sigma2 == "auto":

            d2 = spatial.distance.cdist(centers, centers, 'sqeuclidean')

            d2Max = np.max(d2)

            self.sigma2 = d2Max /(2 * self.Nk)

        else:
            self.sigma2 = sigma2

        self.use_optimized = False
        self.kernel_list = [ "gaussian", "MQ" ,"IMQ" ] #declare optimized implemented kernels

        if self.Kernel in self.kernel_list :

            self.use_optimized = True

            #initialize interpolation matrices lists
            #self.Dx = [None] * self.ndim
            #self.Dxx = [None] * self.ndim
        
        else :
            self.x = sy.symbols("x")
            x =self.x
            sigma2 = self.sigma2
            self.expr = eval(self.Kernel)
        return


       
    def get_interpolation_matrix(self):
        """
        Calculates and returns the interpolation matrix for the given set of points. If `self.use_optimized` is True,
        the optimized version of the interpolation matrix is calculated and returned. Otherwise, the interpolation matrix
        is calculated using a lambda function that evaluates the kernel at every point on the radial function, and then
        returning the resulting matrix after setting all elements with absolute value less than `self.eps` to 0.0 and
        converting the matrix to a sparse CSC format.
        """
        if self.use_optimized :
            G = self.get_interpolation_matrix_optimized( )
        else :

            g = sy.lambdify(self.x, self.expr, "numpy") #create element to element function for evalueate kernel at every point on the radial function

            G = g(self.r2) #evaluate radial function on the kernel

        G[np.abs(G) < self.eps] = 0.0 # take off elements to an specified tolerance
        G = sp.sparse.csc_matrix(G) #transform in csc matrix

        return G

    def get_interpolation_matrix_optimized(self):
        """
        Calculate the interpolation matrix using an optimized method.

        Returns:
        ----------
            G : ndarray 
            The interpolation matrix.
        """
        G = self.kernel(self.r2, self.sigma2, self.Kernel)
        return G

    def get_first_derivative_matrix(self, var_index):
        """
        Calculate the first derivative matrix of a specific variable.

        Parameters
        ----------
            var_index : int 
            The index of the variable for which to calculate the derivative matrix.

        Returns:
        ----------

            ndarray: The first derivative matrix.
        """
        if var_index > self.ndim:
            print("index of variable are higher than ")

        if self.use_optimized:
            Dx = self.get_first_derivative_matrix_optimized(var_index)
        else:
            Dx, _ = self.get_first_derivative_matrix_aux(var_index)

        return Dx


    def get_first_derivative_matrix_optimized(self, var_index):
        """
        Compute the first derivative matrix of the kernel function with respect to the variable at the given index.
        
        Parameters
        ----------
        var_index : int
            The index of the variable to compute the first derivative with respect to.
        
        Returns
        -------
        Dx : scipy.sparse.csc_matrix
            The first derivative matrix.
        
        """
        
        #rx = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: (u - v))
         
        rx = self.points[:, [var_index]] - self.centers[:, var_index]

        Dx = self.kernel_Dx(self.r2, rx, self.sigma2, self.Kernel)

        Dx[np.abs(Dx) < self.eps] = 0.0

        Dx = sp.sparse.csc_matrix(Dx)

        return Dx

       
    def get_first_derivative_matrix_aux(self,var_index):
        """
        Compute the auxiliary matrices needed to compute the first derivative matrix of the kernel function with respect to the variable at the given index.
        
        Parameters
        ----------
        var_index : int
            The index of the variable to compute the first derivative with respect to.
        
        Returns
        -------
        Dx : scipy.sparse.csc_matrix
            The first derivative matrix.
        dr2dx : numpy.ndarray
            The auxiliary array needed to compute the first derivative matrix.
        
        """
        self.gen_f1()

        #dr2dx = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: 2*(u - v))

        dr2dx = 2*(self.points[:, [var_index]] - self.centers[:, var_index])

        #if self.Dx[var_index] == None:

        Dx = self.f1*dr2dx

        Dx[np.abs(Dx) < self.eps] = 0.0


        Dx = sp.sparse.csc_matrix(Dx)

            #self.Dx[var_index] = Dx

        return Dx, dr2dx


    def get_cross_derivative_matrix(self,var_index1,var_index2):
        """
        Compute the cross derivative matrix of the kernel function with respect to the variables at the given indices.
        
        Parameters
        ----------
        var_index1 : int
            The index of the first variable to compute the cross derivative with respect to.
        var_index2 : int
            The index of the second variable to compute the cross derivative with respect to.
        
        Returns
        -------
        Dxy : scipy.sparse.csc_matrix
            The cross derivative matrix.
        
        """
        if var_index1 > self.ndim or var_index2 > self.ndim :
            print("index of variable are higher than dimension")

        if self.use_optimized:

            Dxy = self.get_cross_derivative_matrix_optimized(var_index1,var_index2)

        else :


            d2expr = self.expr.diff(self.x,2)


            d2g = sy.lambdify(self.x, d2expr, "numpy")

            #if self.Dx[var_index]== None :
            _ , dr2dx = self.get_first_derivative_matrix_aux(var_index1)
            _, dr2dy = self.get_first_derivative_matrix_aux(var_index2)


            Dxy = d2g(self.r2)*(dr2dx*dr2dy)+2*self.f1

        Dxy[ np.abs(Dxy) < self.eps] = 0.0


        Dxy = sp.sparse.csc_matrix(Dxy)

        #self.Dxx[var_index] = Dxx

        return Dxy


    def get_cross_derivative_matrix_optimized(self, var_index1,var_index2):
        """
        Compute the cross derivative matrix of the kernel function with respect to the variables at the given indices using an optimized method.
        
        Parameters
        ----------
        var_index1 : int
            The index of the first variable to compute the cross derivative with respect to.
        var_index2 : int
            The index of the second variable to compute the cross derivative with respect to.
        
        Returns
        -------
        Dxy : numpy.ndarray
            The cross derivative matrix.
        
        """
        rx = self.points[:, [var_index1]] - self.centers[:, var_index1]

        ry = self.points[:, [var_index2]] - self.centers[:, var_index2]

        Dxy = self.kernel_Dxy(self.r2,rx,ry ,self.sigma2, self.Kernel)

        return Dxy


    def get_second_derivative_matrix(self,var_index):
        """
        Compute the second derivative matrix of the kernel function with respect to the variable at the given index.
        
        Parameters
        ----------
        var_index : int
            The index of the variable to compute the second derivative with respect to.
        
        Returns
        -------
        Dxx : scipy.sparse.csc_matrix
            The second derivative matrix.
        
        """
        if var_index > self.ndim :
            print("index of variable are higher than dimension")

        if self.use_optimized:

            Dxx = self.get_second_derivative_matrix_optimized(var_index)

        else :


            d2expr = self.expr.diff(self.x,2)


            d2g = sy.lambdify(self.x, d2expr, "numpy")

            #if self.Dx[var_index]== None :
            _ , dr2dx = self.get_first_derivative_matrix_aux(var_index)


            Dxx = d2g(self.r2)*(dr2dx**2)+2*self.f1

        Dxx[ np.abs(Dxx) < self.eps] = 0.0


        Dxx = sp.sparse.csc_matrix(Dxx)

        #self.Dxx[var_index] = Dxx

        return Dxx

    def get_second_derivative_matrix_optimized(self, var_index):
        """
        Compute the second derivative matrix of the kernel function with respect to the variable at the given index using an optimized method.
        
        Parameters
        ----------
        var_index : int
            The index of the variable to compute the second derivative with respect to.
        
        Returns
        -------
        Dxx : numpy.ndarray
            The second derivative matrix.
        
        """
        #rx2 = sp.spatial.distance.cdist(self.points[:, [var_index]], self.centers[:, [var_index]], lambda u, v: (u - v)**2)

        rx2 = (self.points[:, [var_index]] - self.centers[:, var_index])**2

        Dxx = self.kernel_Dxx(self.r2,rx2 ,self.sigma2, self.Kernel)

        return Dxx

    def get_laplacian_matrix(self):
        """
        Compute the Laplacian matrix of the kernel function.
        
        Returns
        -------
        L : scipy.sparse.csc_matrix
            The Laplacian matrix.
        
        """
        if self.use_optimized:
            L = self.kernel_Laplacian(self.r2, self.sigma2, self.Kernel)
        else :

            d2expr = self.expr.diff(self.x,2)


            d2g = sy.lambdify(self.x, d2expr, "numpy")

            self.gen_f1()

            L = d2g(self.r2)*(4*self.r2)+2*self.ndim*self.f1


        L[ np.abs(L) < self.eps ] = 0.0
        L = sp.sparse.csc_matrix(L)

        return L

    def gen_f1(self):
        """
        Generate the first derivative of the kernel function.
        
        """
        if self.f1_was_gen == False:
            d1expr = self.expr.diff(self.x)

            dg = sy.lambdify(self.x, d1expr, "numpy")

            self.f1 = dg(self.r2)

            self.f1_was_gen = True

        return

    def kernel(self,r2,sigma2,Kernel_type):
        """
        Compute the kernel function.
        
        Parameters
        ----------
        r2 : numpy.ndarray
            The array of squared distances.
        sigma2 : float
            The variance parameter of the kernel function.
        Kernel_type : str
            The type of kernel function to use. Can be "gaussian", "MQ", or "IMQ".
        
        Returns
        -------
        G : numpy.ndarray
            The kernel function evaluated at the given squared distances.
        """

        if Kernel_type == "gaussian":
            G = ne.evaluate('exp(-r2/(2.0*sigma2))' )

            #G = np.exp(-r2/(2.0*sigma2))

        elif Kernel_type == "MQ" :

            G = ne.evaluate("sqrt(r2 + sigma2)")

        elif Kernel_type == "IMQ":
            G = ne.evaluate("1.0/sqrt(r2 + sigma2)")

        else :

            print(" this kernel does not exist: ", Kernel_type)

        G = np.float32(G)

        return G

    def kernel_Dx(self,r2,rx,sigma2,Kernel_type):
        """
        Calculate the derivative of a kernel function with respect to a distance value.

        Parameters
        ----------
        r2 : float
            Squared distance between two points.
        rx : float
            Distance between two points in a single dimension.
        sigma2 : float
            Hyperparameter for the kernel function.
        Kernel_type : str
            Type of kernel to use. Must be one of 'gaussian', 'MQ', or 'IMQ'.

        Returns
        -------
        Dx : float
            Derivative of the kernel function with respect to `rx`.

        """
        
        if Kernel_type == "gaussian":
            Dx = ne.evaluate('-(rx/sigma2)*exp(-r2/(2.0*sigma2))')

            #Dx = -(rx/sigma2)*np.exp(-r2/(2.0*sigma2))

        elif Kernel_type == "MQ" :

            Dx = ne.evaluate('rx/sqrt(r2 + sigma2)')

        elif Kernel_type == "IMQ":

            Dx = ne.evaluate('-rx*((r2 + sigma2)**(-1.5))')

        else :
            print(" this kernel does not exist: ",Kernel_type)

        Dx = np.float32(Dx)

        return Dx


    def kernel_Dxy(self,r2,rx,ry,sigma2,Kernel_type):
        """
        Calculate the mixed second partial derivative of a kernel function.

        Parameters
        ----------
        r2 : float
            Squared distance between two points.
        rx : float
            Distance between two points in a single dimension.
        ry : float
            Distance between two points in a single dimension.
        sigma2 : float
            Hyperparameter for the kernel function.
        Kernel_type : str
            Type of kernel to use. Must be one of 'gaussian', 'MQ', or 'IMQ'.

        Returns
        -------
        Dxy : float
            Mixed second partial derivative of the kernel function with respect to `rx` and `ry`.
        """
        if Kernel_type == "gaussian":
            Dxy = ne.evaluate('((rx*ry)/(sigma2**2))*exp(-r2/(2*sigma2))')

        elif Kernel_type == "MQ" :

            Dxy = ne.evaluate('3.0*rx*ry*((r2 + sigma2)**(-2.5))')

        elif Kernel_type == "IMQ":

            Dxy = ne.evaluate('5.0*rx*ry*((r2 + sigma2)**(-3.5))')

        else :
            print(" this kernel does not exist: ",Kernel_type)

        Dxy = np.float32(Dxy)

        return Dxy

    def kernel_Dxx(self,r2,rx2,sigma2,Kernel_type):
        """
        Calculate the Dxx value for a given kernel type.

        Parameters
        ----------
        r2 : float
            The square of the distance between two points.
        rx2 : float
            The square of the distance between two points in the x direction.
        sigma2 : float
            The square of the kernel width.
        Kernel_type : str
            The type of kernel to use. Can be either "gaussian", "MQ", or "IMQ".

        Returns
        -------
        Dxx : float
            The Dxx value for the given kernel type.

        """
        if Kernel_type == "gaussian":

            Dxx = ne.evaluate('((rx2/(sigma2**2)) - (1.0/sigma2) )*exp(-r2/(2.0*sigma2))' )


        elif Kernel_type == "MQ" :

            Dxx = ne.evaluate('(1.0/sqrt(r2 + sigma2))-rx2*((r2 + sigma2)**(-1.5))')
        elif Kernel_type == "IMQ":

            Dxx = ne.evaluate('-((r2 + sigma2)**(-1.5))+3.0*rx2*((r2 + sigma2)**(-2.5))')
        else :
            print(" thins kernel does not exist: ",Kernel_type)

        Dxx = np.float32(Dxx)

        return Dxx

    def kernel_Laplacian(self,r2,sigma2,Kernel_type):
        """
        Calculate the Laplacian value for a given kernel type.

        Parameters
        ----------
        r2 : float
            The square of the distance between two points.
        sigma2 : float
            The square of the kernel width.
        Kernel_type : str
            The type of kernel to use. Can be either "gaussian", "MQ", or "IMQ".

        Returns
        -------
        L : float
            The Laplacian value for the given kernel type.

        """
        ndim = float(self.ndim)

        if Kernel_type == "gaussian":
            L = ne.evaluate('((r2/(sigma2**2.0)) - (ndim/sigma2) )*exp(-r2/(2.0*sigma2))' )
            #L =((r2 / (sigma2 ** 2.0)) - (ndim / sigma2)) * np.exp(-r2 / (2.0 * sigma2))

        elif Kernel_type == "MQ" :
            L = ne.evaluate('(ndim/sqrt(r2 + sigma2))-r2*((r2 + sigma2)**(-1.5))')
            #L = 2/np.sqrt(r2 + (2 * sigma2)) - r2*np.power(r2 + (2 * sigma2),1.5)

        elif Kernel_type == "IMQ":
            L = ne.evaluate('-ndim*((r2 + sigma2)**(-1.5))+3.0*r2*((r2 + sigma2)**(-2.5))')
            #L = -2*np.power(r2 + (2 * sigma2),-1.5)+3*r2*np.power(r2 + (2 * sigma2),-2.5)
        else :
            print(" thins kernel does not exist: ",Kernel_type)

        L = np.float32(L)

        return L