%Run ML_assign5_2_5_3.m first.
k = 1:10;
plot(k,p1,k,p2,k,p3,'-', 'LineWidth', 3);
grid;
xlabel 'No. of Clusters(k)'
ylabel 'p1,p2,p3'
