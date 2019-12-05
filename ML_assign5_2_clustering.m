function [SS, centroids, ind] = ML_assign5_2_clustering(X, k, rand_cent)
    X_size = size(X);
    fprintf('\nSize of cluster, k= %d',k);
    centroids = zeros(k,size(X,2));
    
    if rand_cent
        randidx = randperm(size(X,1));
        centroids = X(randidx(1:k), :); 
    else
        centroids = X(1:k,1:X_size(2));
    end
    
    ind = zeros(X_size(1),1);
    prev_ind = zeros(X_size(1),1);
    SS = 0;

    for i = 1:20
        % clustering
        for points = 1:X_size(1)
            sp = 1;
            dist_min = sqrt(sum((X(points,1:X_size(2))- centroids (1,1:X_size(2))).^2));
            for center = 2:k
               dist = sqrt(sum((X(points,1:X_size(2))- centroids (center,1:X_size(2))).^2));
               if (dist_min > dist)
                   dist_min = dist;
                   sp = center;
               end
            end
            ind(points) = sp;
        end

        % Recalculating Centroids

        for j = 1:k
            m = X(ind == j,1:X_size(2));
            centroids (j,1:X_size(2)) = mean(m);
            %SS = SS + sum(sum((m - mean(m)).^2));
        end

        %Checking the change in centroids
        
        if  isequal(prev_ind,ind)
            %fprintf('\nbreaking...');
            break;
        else
            prev_ind = ind;
        end
    end
    fprintf('\nExiting on iteration = %d\n',i);
    
    for j = 1:k
        m = X(ind == j,1:X_size(2));
        centroids (j,1:X_size(2)) = mean(m);
        SS = SS + sum(sum((m - mean(m)).^2));
    end
    centroids = centroids';
    
    
   %fprintf('total within group sum of squares = %d\n',SS);
    %gscatter(X,Y,ind);
end
    