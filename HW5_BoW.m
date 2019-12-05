classdef HW5_BoW    
% Practical for Visual Bag-of-Words representation    
% Use SVM, instead of Least-Squares SVM as for MPP_BoW
% By: Minh Hoai Nguyen (minhhoai@cs.stonybrook.edu)
% Created: 18-Dec-2015
% Last modified: 16-Oct-2018    
    
    methods (Static)
        function main()
            scales = [8, 16, 32, 64];
            normH = 16;
            normW = 16;
            bowCs = HW5_BoW.learnDictionary(scales, normH, normW);
            
            [trIds, trLbs] = ml_load('C:\Users\rohit\Downloads\ML\Assignment 5\hw5data\hw5data\bigbangtheory_v3\train.mat',  'imIds', 'lbs');             
            tstIds = ml_load('C:\Users\rohit\Downloads\ML\Assignment 5\hw5data\hw5data\bigbangtheory_v3\test.mat', 'imIds'); 
            
            features = sprintf('C:/Users/rohit/Downloads/ML/Assignment 5/hw5data/hw5data/src/features.mat');
            %features2 = sprintf('C:/Users/rohit/Downloads/ML/Assignment 5/hw5data/hw5data/src/features2.mat');
            if exist(features, 'file')
                load(features);
            else
                trD  = HW5_BoW.cmpFeatVecs(trIds, scales, normH, normW, bowCs);
                tstD = HW5_BoW.cmpFeatVecs(tstIds, scales, normH, normW, bowCs);
                save(features, 'trD','tstD');
                %save(features2, 'trD','tstD');
            end            

            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Write code for training svm and prediction here            %
             trD = trD';
             tstD = tstD';

            
            %3.4.2
            fprintf('\nQ 3.4.2 SVM and RBF kernel (Default params),\n')
            CV_Accuracy1 = svmtrain(trLbs,trD,'-q -t 2 -v 5');
            fprintf('\nUsed  default C ,gamma and got  Cross Validation Accuracy = %f%',CV_Accuracy1);

           
            %3.4.3
            fprintf('\nQ 3.4.3 SVM and RBF kernel (Tuned params),\n')
            bestcv = 0;
            for log2c = 8:10
              for log2g = 4:6
                cmd = ['-q -t 2 -v 5 -c ', num2str(2^log2c), ' -g ', num2str(2^log2g)];
                cv = svmtrain(trLbs, trD, cmd);
                if (cv >= bestcv)
                  bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
                end
                fprintf('%g %g %g (best C=%g, gamma=%g, Cross Validation Accuracy =%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
              end
            end
             
             
            %3.4.5
            fprintf('\n\n3.4.5 SVM and chi^2 kernel (Tuned params),\n')
            bestcv = 0;
            for log2c = 9:10
              for log2g = 0:1
                cmd = ['-q -t 4 -v 5 -c ', num2str(2^log2c)];
                [trainK, testK] = cmpExpX2Kernel(trD, tstD, 2^log2g);
                cv = svmtrain(trLbs, [transpose(1:length(trD')), trainK], cmd); 
                if (cv >= bestcv)
                  bestcv = cv; bestc = 2^log2c; bestg = 2^log2g;
                end
                fprintf('%g %g %g (best C=%g, gamma=%g, Cross Validation Accuracy =%g)\n', log2c, log2g, cv, bestc, bestg, bestcv);
              end
            end            
            %cal gamma
%               fprintf('\n3.4.5 SVM and chi^2 kernel (Tuned params),\n')
% %               %gamma3 = calculate_gamma(trD);
%                gamma3 =1.1051709;
%                cost3 = 2.6e+43;     
% %               save('tr.mat','trD', 'tstD', 'trLbs');
%                [trainK, testK] = cmpExpX2Kernel(trD, tstD, gamma3);
% %               %[trainK, testK] = cmpExpX2Kernel(trD, tstD, 1.00100050017);  
% %               [trainK, testK] = cmpExpX2Kernel(trD, tstD, 0.1000001);                            
%                save('beforepred.mat','trainK','testK','tstD', 'trD','trLbs','tstD');
%                cmd = ['-q -t 4 -v 5 -c ', num2str(cost3)];
%                CV_Accuracy2 = svmtrain(trLbs, [transpose(1:length(trD')), trainK], cmd ); 
% %               %CV_Accuracy2 = svmtrain(trLbs, [transpose(1:length(trD')), trainK], '-q -t 4 -c 22026.4657948 -v 5');
% %               %CV_Accuracy2 = svmtrain(trLbs, [transpose(1:length(trD')), trainK], '-q -t 4 -c 22026.4657948 -v 5');
%                fprintf('\nUsed C = %f,gamma = %f and got Accuracy = %f',cost3, gamma3, CV_Accuracy2);
%             
             %3.4.6
            fprintf('\n3.4.6 Predicting Test Labels,\n');
            %gamma3 = calculate_gamma(trD);
            [trainK, testK] = cmpExpX2Kernel(trD, tstD, 1.1051709);
            %cmd = ['-q -t 4 -v 5 -c ', num2str(2^10)];
            %cv = svmtrain(trLbs, [transpose(1:length(trD')), trainK], cmd); 
            %fprintf('CV accuracy : %f %',cv);
            cmd2 = ['-q -t 4 -c ', num2str(2.6e+43)];
            model = svmtrain(trLbs, [transpose(1:length(trD')), trainK], cmd2);
            testlbs = zeros(size(tstD,1),1);
            %save('beforepred.mat','model', 'trainK','testK','tstD', 'trD');
            [predicted_label] = svmpredict(testlbs,[transpose(1:length(tstD')), testK'], model);
            %save('predictlb.mat', 'predicted_label');
            pred = horzcat(transpose(1:length(tstD')),predicted_label);
            %save('predict.mat', 'predicted_label','pred');             
            T = array2table(pred);
            T.Properties.VariableNames(1:2) = {'ImgId','Prediction'};
            writetable(T,'predTestLabels.csv')
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            
        end
                
        function bowCs = learnDictionary(scales, normH, normW)
            % Number of random patches to build a visual dictionary
            % Should be around 1 million for a robust result
            % We set to a small number her to speed up the process. 
            nPatch2Sample = 100000;
            
            % load train ids
            trIds = ml_load('C:\Users\rohit\Downloads\ML\Assignment 5\hw5data\hw5data\bigbangtheory_v3\train.mat', 'imIds'); 
            nPatchPerImScale = ceil(nPatch2Sample/length(trIds)/length(scales));
                        
            randWins = cell(length(scales), length(trIds)); % to store random patches
            for i=1:length(trIds);
                ml_progressBar(i, length(trIds), 'Randomly sample image patches');
                im = imread(sprintf('C:/Users/rohit/Downloads/ML/Assignment 5/hw5data/hw5data/bigbangtheory_v3/%06d.jpg', trIds(i)));
                im = double(rgb2gray(im));  
                for j=1:length(scales)
                    scale = scales(j);
                    winSz = [scale, scale];
                    stepSz = winSz/2; % stepSz is set to half the window size here. 
                    
                    % ML_SlideWin is a class for efficient sliding window 
                    swObj = ML_SlideWin(im, winSz, stepSz);
                    
                    % Randomly sample some patches
                    randWins_ji = swObj.getRandomSamples(nPatchPerImScale);
                    
                    % resize all the patches to have a standard size
                    randWins_ji = reshape(randWins_ji, [scale, scale, size(randWins_ji,2)]);                    
                    randWins{j,i} = imresize(randWins_ji, [normH, normW]);
                end
            end
            randWins = cat(3, randWins{:});
            randWins = reshape(randWins, [normH*normW, size(randWins,3)]);
                                    
            fprintf('Learn a visual dictionary using k-means\n');
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Use your K-means implementation here                       %
            % to learn visual vocabulary                                 %
            % Input: randWins contains your data points                 %
            cacheFile = sprintf('C:/Users/rohit/Downloads/ML/Assignment 5/hw5data/hw5data/src/kmeans_bow.mat');
            if exist(cacheFile, 'file')
                load(cacheFile);
            else
                [~, bowCs, ~] = ML_assign5_2_clustering(randWins',1000,1);
              %[~, bowCs, ~] = ML_assign5_2_clustering(randWins',1,1);

                save(cacheFile, 'bowCs');
            end
            %disp(bowCs);
            % Output: bowCs: centroids from k-means, one column for each centroid  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        end
                
        function D = cmpFeatVecs(imIds, scales, normH, normW, bowCs)
            n = length(imIds);
            D = cell(1, n);
            startT = tic;
            for i=1:n
                ml_progressBar(i, n, 'Computing feature vectors', startT);
                im = imread(sprintf('C:/Users/rohit/Downloads/ML/Assignment 5/hw5data/hw5data/bigbangtheory_v3/%06d.jpg', imIds(i)));                                
                bowIds = HW5_BoW.cmpBowIds(im, scales, normH, normW, bowCs);                
                feat = hist(bowIds, 1:size(bowCs,2));
                D{i} = feat(:);
            end
            D = cat(2, D{:});
            D = double(D);
            D = D./repmat(sum(D,1), size(D,1),1);
            
        end        
        
        % bowCs: d*k matrix, with d = normH*normW, k: number of clusters
        % scales: sizes to densely extract the patches. 
        % normH, normW: normalized height and width oMf patches
        function bowIds = cmpBowIds(im, scales, normH, normW, bowCs)
            im = double(rgb2gray(im));
            bowIds = cell(length(scales),1);
            for j=1:length(scales)
                scale = scales(j);
                winSz = [scale, scale];
                stepSz = winSz/2; % stepSz is set to half the window size here.
                
                % ML_SlideWin is a class for efficient sliding window
                swObj = ML_SlideWin(im, winSz, stepSz);
                nBatch = swObj.getNBatch();
                
                for u=1:nBatch
                    wins = swObj.getBatch(u);
                    
                    % resize all the patches to have a standard size
                    wins = reshape(wins, [scale, scale, size(wins,2)]);                    
                    wins = imresize(wins, [normH, normW]);
                    wins = reshape(wins, [normH*normW, size(wins,3)]);
                    
                    % Get squared distance between windows and centroids
                    dist2 = ml_sqrDist(bowCs, wins); % dist2: k*n matrix, 
                    
                    % bowId: is the index of cluster closest to a patch
                    [~, bowIds{j,u}] = min(dist2, [], 1);                     
                end                
            end
            bowIds = cat(2, bowIds{:});
        end        
        
    end    
end

