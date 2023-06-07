import copy
import tensorly as tl
import numpy as np
from helpers import *


def cmtf4si(corrupt_img, y=None, c_m=None, r=10, omega=None, alpha=1, tol=1e-4, maxiter=800, init='random', printitn=1000):
    """
    CMTF Compute a Coupled Matrix and Tensor Factorization (and recover the Tensor).
    ---------
    :param   'corrupt_img'  - Tensor
    :param   'y'  - Coupled Matries
    :param  'c_m' - Coupled Modes
    :param   'r'  - Tensor Rank
    :param  'omega'- Index Tensor of Obseved Entries
    :param 'alpha'- Impact factor for HiL part {0.0-1.0}
    :param  'tol' - Tolerance on difference in fit {1.0e-4}
    :param 'maxiters' - Maximum number of iterations {50}
    :param 'init' - Initial guess [{'random'}|'nvecs'|cell array]
    :param 'printitn' - Print fit every n iterations; 0 for no printing {1}
    ---------
    :return
     P: Decompose result.(kensor)
     x: Recovered Tensor.
     V: Projection Matrix.
    ---------
    """
    x = corrupt_img.copy()
    # Construct omega if no input
    if omega is None:
        omega = x * 0 + 1
    bool_omeg = np.array(omega, dtype=bool)
    # Extract number of dimensions and norm of x.
    N = len(x.shape)
    normX = np.linalg.norm(x)
    dimorder = np.arange(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
    maxiters = maxiter

    # Recover or just decomposition
    recover = 0
    if 0 in omega:
        recover = 1

    Uinit = []
    Uinit.append([])
    for n in dimorder[1:]:
        Uinit.append(np.random.random([x.shape[n], r]))
        
    # Set up for iterations - initializing U and the fit.
    # STEP 1: a random V is initialized - y x V =  |mode x rank|
    U = Uinit[:]
    if type(c_m) == int:
        V = np.random.random([y.shape[1], r])
    else:
        V = [np.random.random([y[i].shape[1], r]) for i in range(len(c_m))]
    fit = 0

    # Save hadamard product of each U[n].T*U[n]
    UtU = np.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = np.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = x * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = X[n]* khatrirao(all U except n, 'r').
            ktr = tl.tenalg.khatri_rao(U, weights=None, skip_matrix=n)
            Unew = np.dot(tl.unfold(x, n) ,ktr)

            # Compute the matrix of coefficients for linear system
            temp = list(range(n))
            temp[len(temp):len(temp)] = list(range(n + 1, N))
            B = np.prod(UtU[temp, :, :], axis=0)
            if int != type(c_m):
                tempCM = [i for i, a in enumerate(c_m) if a == n]
            elif c_m == n:
                tempCM = [0]
            else:
                tempCM = []
            if tempCM != [] and int != type(c_m):
                for i in tempCM:
                    B = B + np.dot(V[i].T, V[i])
                    Unew = Unew + np.dot(y[i], V[i])
                    V[i] = np.dot(y[i].T, Unew)
                    V[i] = V[i].dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            elif tempCM != []:
                B = B + np.dot(V.T, V)
                Unew = Unew + np.dot((alpha)*(y*~bool_omeg)+(y*bool_omeg), V)
                V = np.dot(((alpha)*(y*~bool_omeg)+(y*bool_omeg)).T, Unew)
                V = V.dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            Unew = Unew.dot(np.linalg.inv(B))
            U[n] = Unew
            UtU[n, :, :] = np.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        lamb = np.ones(r)
        final_shape = tuple(u.shape[0] for u in U)
        P = np.dot(lamb.T, tl.tenalg.khatri_rao(U).T)
        P = P.reshape(final_shape)
        x[bool_omeg] = corrupt_img[bool_omeg]
        x[~bool_omeg] = P[~bool_omeg]

        fitchange = np.linalg.norm(x - oldX)


        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print ('CMTF: iterations=',iter, 'f=',fit, 'f-delta=',fitchange)
            else:
                print ('CMTF: iterations=',iter, 'f-delta=',fitchange)
        if flag == 0:
            break

    return P, x, V

def cmtf(corrupt_img, y=None, c_m=None, r=20, omega=None, tol=1e-4, maxiter=800, init='random', printitn=500):
    """
    CMTF Compute a Coupled Matrix and Tensor Factorization (and recover the Tensor).
    ---------
    :param   'corrupt_img'  - Tensor
    :param   'y'  - Coupled Matrices
    :param  'c_m' - Coupled Modes
    :param   'r'  - Tensor Rank
    :param  'omega'- Index Tensor of Observed Entries (mask)
    :param  'tol' - Tolerance on difference in fit {1.0e-4}
    :param 'maxiters' - Maximum number of iterations {50}
    :param 'init' - Initial guess [{'random'}|'nvecs'|cell array]
    :param 'printitn' - Print fit every n iterations; 0 for no printing {1}
    ---------
    :return
     P: Decompose result.(kensor)
     x: Recovered Tensor.
    #  V: Projection Matrix.
    ---------
    """
    print('CMTF:')
    x = copy.deepcopy(corrupt_img)

    if c_m is None:
        c_m = 0
    elif int == type(c_m):
        c_m = c_m - 1
    else:
        c_m = [i - 1 for i in c_m]

    # Construct omega if no input
    if omega is None:
        omega = x * 0 + 1
    
    bool_omeg = np.array(omega, dtype=bool)

    # Extract number of dimensions and norm of x.
    N = len(x.shape)
    normX = np.linalg.norm(x)
    dimorder = np.arange(N) #[0,1,2] for N=3 # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = 1e-4
    maxiters = maxiter

    # Recover or just decomposition
    recover = 0
    if 0 in omega:
        recover = 1

    Uinit = []
    Uinit.append([])
    for n in dimorder[1:]:
        Uinit.append(np.random.random([x.shape[n], r]))
        
    # Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    if type(c_m) == int:
        V = np.random.random([y.shape[1], r])
    else:
        V = [np.random.random([y[i].shape[1], r]) for i in range(len(c_m))]
    fit = 0

    # Save hadamard product of each U[n].T*U[n]
    UtU = np.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = np.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = x * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
             # Calculate Unew = X[n]* khatrirao(all U except n, 'r').
            ktr = tl.tenalg.khatri_rao(U, weights=None, skip_matrix=n) #000
            Unew = np.dot(tl.unfold(x, n) ,ktr) #000

            # Compute the matrix of coefficients for linear system
            temp = list(range(n))
            temp[len(temp):len(temp)] = list(range(n + 1, N))
            B = np.prod(UtU[temp, :, :], axis=0)
            if int != type(c_m):
                tempCM = [i for i, a in enumerate(c_m) if a == n]
            elif c_m == n:
                tempCM = [0]
            else:
                tempCM = []
            if tempCM != [] and int != type(c_m):
                for i in tempCM:
                    B = B + np.dot(V[i].T, V[i])
                    Unew = Unew + np.dot(y[i], V[i])
                    V[i] = np.dot(y[i].T, Unew)
                    V[i] = V[i].dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            elif tempCM != []:
                B = B + np.dot(V.T, V)
                tempvt = np.dot(y, V)
                Unew = Unew + tempvt
                V = np.dot(y.T, Unew)
                V = V.dot(np.linalg.inv(np.dot(Unew.T, Unew)))
            Unew = Unew.dot(np.linalg.inv(B))
            U[n] = Unew
            UtU[n, :, :] = np.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        lamb = np.ones(r)
        # P = pyten.tenclass.Ktensor(lamb, U)
        final_shape = tuple(u.shape[0] for u in U)
        P = np.dot(lamb.T, tl.tenalg.khatri_rao(U).T)
        P = P.reshape(final_shape)
        x[bool_omeg] = corrupt_img[bool_omeg]
        x[~bool_omeg] = P[~bool_omeg]
        # x, _ = replacePixels(x, np.ravel(~bool_omeg), P)

        fitchange = np.linalg.norm(x - oldX)


        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            if recover == 0:
                print ('CMTF: iterations=',iter, 'f=',fit, 'f-delta=',fitchange)
            else:
                print ('CMTF: iterations=',iter, 'f-delta=',fitchange)
        if flag == 0:
            break

    return P, x

def cp_als(y, r=12, omega=None, tol=1e-4, maxiter=800, init='random', printitn=500, original_img = None):
    """ CP_ALS Compute a CP decomposition of a Tensor (and recover it).
    ---------
     :param  'y' - Tensor with Missing data
     :param  'r' - Rank of the tensor
     :param 'omega' - Missing data Index Tensor
     :param 'tol' - Tolerance on difference in fit
     :param 'maxiters' - Maximum number of iterations
     :param 'init' - Initial guess ['random'|'nvecs'|'eigs']
     :param 'printitn' - Print fit every n iterations; 0 for no printing
    ---------
     :return
        'P' - Decompose result.(kensor)
        'X' - Recovered Tensor.
    ---------
    """
    print('cp_als:')
    X = copy.deepcopy(y)

    if omega is None:
        omega = X.data * 0 + 1
    
    bool_omeg = np.array(omega, dtype=bool)


    # Extract number of dimensions and norm of X.
    N = len(X.shape)
    normX = np.linalg.norm(X)
    dimorder = np.arange(N)  # 'dimorder' - Order to loop through dimensions {0:(ndims(A)-1)}

    # Define convergence tolerance & maximum iteration
    fitchangetol = tol
    maxiters = maxiter

    Uinit = []
    Uinit.append([])
    for n in dimorder[1:]:
        Uinit.append(np.random.random([X.shape[n], r]))


    # Set up for iterations - initializing U and the fit.
    U = Uinit[:]
    fit = 0

    if printitn > 0:
        print('\nCP_ALS:\n')

    # Save hadamard product of each U[n].T*U[n]
    UtU = np.zeros([N, r, r])
    for n in range(N):
        if len(U[n]):
            UtU[n, :, :] = np.dot(U[n].T, U[n])

    for iter in range(1, maxiters + 1):
        fitold = fit
        oldX = X * 1.0

        # Iterate over all N modes of the Tensor
        for n in range(N):
            # Calculate Unew = corrupt_img(n) * khatrirao(all U except n, 'r').
            ktr = tl.tenalg.khatri_rao(U, weights=None, skip_matrix=n)
            Unew = np.dot(tl.unfold(X, n) ,ktr)

            # Compute the matrix of coefficients for linear system
            temp = list(range(n))
            temp[len(temp):len(temp)] = list(range(n + 1, N))
            B = np.prod(UtU[temp, :, :], axis=0)
            Unew = Unew.dot(np.linalg.inv(B))

            # Normalize each vector to prevent singularities in coefmatrix
            if iter == 1:
                lamb = np.sqrt(np.sum(np.square(Unew), 0))  # 2-norm
            else:
                lamb = np.max(Unew, 0)
                lamb = np.max([lamb, np.ones(r)], 0)  # max-norm

            lamb = np.array([x * 1.0 for x in lamb])
            Unew = Unew / lamb
            U[n] = Unew
            UtU[n, :, :] = np.dot(U[n].T, U[n])

        # Reconstructed fitted Ktensor
        final_shape = tuple(u.shape[0] for u in U)
        P = np.dot(lamb.T, tl.tenalg.khatri_rao(U).T)
        P = P.reshape(final_shape)
        X[bool_omeg] = y[bool_omeg]
        X[~bool_omeg] = P[~bool_omeg]
        # X[~bool_omeg] = original_img[~bool_omeg]
        # X, _ = replacePixels(X, np.ravel(~bool_omeg), P)

        fitchange = np.linalg.norm(X - oldX)

        # Check for convergence
        if (iter > 1) and (fitchange < fitchangetol):
            flag = 0
        else:
            flag = 1

        if (printitn != 0 and iter % printitn == 0) or ((printitn > 0) and (flag == 0)):
            print ('cp_als: iterations=',iter, 'f-delta=',fitchange)

        # Check for convergence
        if flag == 0:
            break

    return P, X

def siLRTC(corrupt_imgorig, mask2D, a, b, K):
    print('SiLRTC:')
    bool_omeg = np.array(mask2D, dtype=bool)

    corrupt_img = copy.deepcopy(corrupt_imgorig)
    orig = copy.deepcopy(corrupt_imgorig)
    imSize = corrupt_img.shape
    for _ in range(K):
        print(_, end=" ")
        M = np.zeros(imSize)
        for j in range(3):
            unf = b[j]*shrinkage(tl.base.unfold(corrupt_img, j),a[j]/b[j])
            M = M + tl.fold(unf, j, imSize)
        M/=sum(b)

        # Update indices that we know from Image into M and set corrupt_img equal to M
        M[bool_omeg] = orig[bool_omeg]
        corrupt_img = M
    return corrupt_img

def haLRTC(corrupt_imgorig, mask2D, a, b, K):

    corrupt_img = copy.deepcopy(corrupt_imgorig)
    orig = copy.deepcopy(corrupt_imgorig)
    print('HaLRTC:')
    bool_omeg = np.array(mask2D, dtype=bool)
    p = 1e-6
    imSize = corrupt_img.shape
    ArrSize = np.array(imSize)
    ArrSize = np.append(ArrSize, 3) # array of Mi / Yi
    Mi = np.zeros(ArrSize)
    Yi = np.zeros(ArrSize)

    for _ in range(K):
        print(_, end=" ")
        for i in range(ArrSize[3]):
            elem = tl.unfold(corrupt_img, i)
            elem = np.add(elem,tl.unfold(np.squeeze(Yi[:, :, :, i]), i) / p,out = elem, casting="unsafe")
            elem = shrinkage(elem, a[i] / p)
            Mi[:,:,:,i] = tl.fold(elem, i, imSize)
        
        corrupt_img = (1/ArrSize[3]) * np.sum(Mi - Yi/p, ArrSize[3])
        corrupt_img[bool_omeg] = orig[bool_omeg]

        for i in range(ArrSize[3]):
            Yi[:, :, :, i] = Yi[:, :, :, i] - p * (Mi[:, :, :, i] - corrupt_img)
        
        p = 1.2 * p


    return corrupt_img

# def faLRTC(corrupt_imgorig, mask2D, a, b, K, img):
#     print('FaLRTC:')
#     corrupt_img = copy.deepcopy(corrupt_imgorig)
#     imSize = corrupt_img.shape
#     C = 0.5
#     u = 10^5
#     ui = a/u
#     Z = copy.deepcopy(corrupt_img)
#     W = copy.deepcopy(corrupt_img)
#     B = 0
#     L = np.sum(ui)
#     for _ in range(K):
#         print(_, end=" ")
#         while True:
#             theta = (1 + np.sqrt(1 + 4 * L * B)) / (2 * L)
#             thetabyL = theta/L
#             W = thetabyL/(B + thetabyL) * Z + B/(B+thetabyL) * corrupt_img

#             dfw = np.zeros(imSize)
#             fx = 0
#             fw = 0

#             for i in range(3):
#                 Sig, T = Trimming(tl.unfold(corrupt_img,i), ui[i]/a[i])
#                 fx += np.sum(Sig)
#                 Sig, T = Trimming(tl.unfold(W,i), ui[i]/a[i])
#                 fw += np.sum(Sig)
#                 dfw += tl.fold(a[i]**2/ui[i]*T, i, imSize)
            
#             # print(dfw.shape)
#             dfw = np.array(np.gradient(dfw, axis = 0))
#             # print(dfw.shape)
            
#             # print(fx)
#             # print(fw - np.sum(dfw**2)/L)
#             if fx <= fw - np.sum(dfw**2)/(2*L):
#                 # print('break1')
#                 break

#             Xp = W - dfw/L
#             fxp = 0
#             for i in range(3):
#                 Sig, T = Trimming(tl.unfold(Xp,i), ui[i]/a[i])
#                 fxp += np.sum(Sig)
            
#             # print(fxp)
#             # print(fw - np.sum(dfw**2)/(2*L))
#             if fxp-20 <= fw - np.sum(dfw**2)/(2*L):
#                 corrupt_img[mask2D] = Xp[mask2D]
#                 # print('break2')
#                 break
            
#             L = L/C

#             # print(L)
#             if L > 1e10:
#                 print("L too large, ending function..")
#                 return corrupt_img

#         Z = np.subtract(Z,thetabyL*dfw, out = Z, casting="unsafe") #check uint8/float64 issue
#         B += thetabyL


#     return corrupt_img