X = load('C:\Users\rohit\Downloads\ML\Assignment 5\hw5data\hw5data\digit\digit.txt');
Y = load('C:\Users\rohit\Downloads\ML\Assignment 5\hw5data\hw5data\digit\labels.txt');
fprintf('\n');

rand_cent =0;

k = 2;
[SS2, centroids2, ind2] = ML_assign5_2_clustering(X, k, rand_cent);
fprintf('total within group sum of squares = %d\n',SS2);
[p1, p2, p3] = pair_counting_measures(X, Y, ind2);
fprintf('p1 = %f, p2 = %f, p3 = %f',p1,p2,p3);
k = 4;
[SS4, centroids4, ind4] = ML_assign5_2_clustering(X, k, rand_cent);
fprintf('total within group sum of squares = %d\n',SS4);
[p1, p2, p3] = pair_counting_measures(X, Y, ind4);
fprintf('p1 = %f, p2 = %f, p3 = %f',p1,p2,p3);
k = 6;
[SS6, centroids6, ind6] = ML_assign5_2_clustering(X, k, rand_cent);
fprintf('total within group sum of squares = %d\n',SS6);
[p1, p2, p3] = pair_counting_measures(X, Y, ind6);
fprintf('p1 = %f, p2 = %f, p3 = %f',p1,p2,p3);
