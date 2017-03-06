% Demo of EAGR
% Detials in 'Scalable Semi-Supervised Learning by Efficient Anchor Graph Regularization'
clear
load('Data_Letter.mat')
[IDX,anchor]=kmeans(data,500,'MaxIter',10,'emptyaction','singleton');
[Z,rL] = FLAE(anchor,data,3,1);
acc=zeros(1,5);
for iter=1:5
    [F, A, err] = AnchorGraphReg(Z,rL, label', label_index(iter,:), 1);
    acc(iter) =1-err;   
end
fprintf('\n The average classification accuracy of EAGR is %.2f%%.\n', mean(acc)*100);