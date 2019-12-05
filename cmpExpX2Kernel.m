
function [trainK, testK] = cmpExpX2Kernel(trainD, testD, gamma)
    train_size = size(trainD);
    test_size = size(testD);
    trainK = zeros(train_size(1),train_size(1));
    testK = zeros(test_size(1),test_size(1));
    epsilon = 0.0001;
    for i = 1:train_size(1)
        for j = 1: train_size(1)
            t1 = (trainD(i,:) - trainD(j,:)).^2;
            t2 = trainD(i,:) + trainD(j,:)+ epsilon;
            trainK(i,j) = exp(sum(t1./t2)*(-1/gamma));
        end
    end
    for i = 1:train_size(1)
        for j = 1: test_size(1)
            t1 = (trainD(i,:) - testD(j,:)).^2;
            t2 = trainD(i,:) + testD(j,:)+ epsilon;
            testK(i,j) = exp(sum(t1./t2)*(-1/gamma));
        end
    end
    
end