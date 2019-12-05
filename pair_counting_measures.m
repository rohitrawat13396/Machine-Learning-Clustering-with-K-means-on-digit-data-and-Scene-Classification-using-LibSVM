function [p1, p2, p3] = pair_counting_measures(X, Y, ind)
    X_size = size(X);
    p1 = 0;
    p2 = 0;
    p3 = 0;
    t1 = 0;
    t2 = 0;
    for p = 1:X_size(1)
        for q = 1:X_size(1)
            if Y(p) == Y(q)
                t1 = t1+1;
                if ind(p)==ind(q)
                    p1 = p1 + 1;
                end
            end
            if Y(p) ~= Y(q)
                t2 = t2+1;
                if ind(p)~=ind(q)
                    p2 = p2 + 1;
                end
            end
        end
    end
   p1 = (p1/(t1))*100;
   p2 = (p2/(t2))*100;
   p3 = (p1+p2)/2;
end