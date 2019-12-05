TSS = zeros(10,1);
p1 = zeros(10,1);
p2 = zeros(10,1);
p3 = zeros(10,1);

for k = 1:10
    for ten = 1:10
        [SS, centroids, ind] = ML_assign5_2_clustering(X, k, 1);
        TSS(k,1) = TSS(k,1) + SS;
        [pp1, pp2, pp3] = pair_counting_measures(X, Y, ind);
        p1(k,1) = p1(k,1) + pp1;
        p2(k,1) = p2(k,1) + pp2;
        p3(k,1) = p3(k,1) + pp3;
    end
end
TSS = TSS/10;
p1 = p1/10;
p2 = p2/10;
p3 = p3/10;

k = 1:10;
plot(k,TSS,'-', 'LineWidth', 3);
grid;
xlabel 'No. of Clusters(k)'
ylabel 'SS'

