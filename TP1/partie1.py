import numpy as np

def main():
    # pluie = P(P=0) = 0.8
    #         P(P=1)   0.2
    pluie = np.array([0.8, 0.2]).reshape(2, 1, 1, 1)
    print('Pr(Pluie)={}\n'.format(np.squeeze(pluie)))

    # arroseur = P(A=0) = 0.9
    #            P(A=1)   0.1
    arroseur = np.array([0.9, 0.1]).reshape(1, 2, 1, 1)
    print('Pr(Arroseur)={}\n'.format(np.squeeze(arroseur)))


    # watson = P(W=0|P=0)  P(W=1|P=0) = 0.8  0.2
    #          P(W=0|P=1)  P(W=1|P=1)   0    1
    watson = np.array([[0.8, 0.2], [0, 1]]).reshape(2, 1, 2, 1)
    print('Pr(Watson|Pluie)={}\n'.format(np.squeeze(watson)))

    # holmes = P(H=0|P=0, A=0)  P(H=1|P=0, A=0) =   1    0
    #          P(H=0|P=1, A=0)  P(H=1|P=1, A=0)     0.1  0.9
    #          P(H=0|P=0, A=1)  P(H=1|P=0, A=1)     0    1
    #          P(H=0|P=1, A=1)  P(H=1|P=1, A=1)     0    1
    holmes = np.array([[1, 0], [0.1, 0.9], [0, 1], [0, 1]]).reshape(2, 2, 1, 2)
    print('Pr(Holmes|Pluie, Arroseur)={}\n'.format(np.squeeze(holmes)))

    prob_conjointe = pluie * arroseur * holmes * watson

    # A. Pr(W=1)
    A = prob_conjointe[:, :, 1, :].sum()
    print('A. Pr(W=1) = {}'.format(A))

    # B. Pr(W=1|H=1)
    B = prob_conjointe[:, :, 1, 1].sum()/prob_conjointe[:, :, :, 1].sum()
    print('B. Pr(W=1|H=1) = {}'.format(B))

    # C. Pr(W=1|H=1, A=0)
    C = prob_conjointe[:, 0, 1, 1].sum()/prob_conjointe[:, 0, :, 1].sum()
    print('C. Pr(W=1|H=1,A=0) = {}'.format(C))

    # D. Pr(W=1|A=0)=Pr(W=1)
    D = prob_conjointe[:, 0, 1, :].sum()/prob_conjointe[:, 0, :, :].sum()
    print('D. Pr(W=1|A=0) = {}'.format(D))

    # E. Pr(W=1|P=1)
    E = prob_conjointe[1, :, 1, :].sum()/prob_conjointe[1, :, :, :].sum()
    print('E. Pr(W=1|P=1) = {}'.format(E))

if __name__ == '__main__':
    main()
