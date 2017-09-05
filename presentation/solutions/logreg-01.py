W_list = [W0]
b_list = [b0]
for i in range(200):       # 200 steps of gradient descend
    Wold = W_list[-1]
    bold = b_list[-1]
    Wnew = Wold - eta*logreg_gradW(X,y,Wold, bold )
    bnew = bold - eta*logreg_gradb(X,y,Wold, bold )
    W_list.append(Wnew)
    b_list.append(bnew)
    
W_list = np.stack(W_list)[:,:,0]

plt.plot(W_list);
plt.plot(b_list);
plt.xlabel('Epoch')
plt.ylabel('Parameter')
plt.legend(['W0', 'W1', 'b'])