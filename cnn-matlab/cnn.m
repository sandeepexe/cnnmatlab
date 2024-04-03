% Load the dataset
rootFolder = 'path/to/caltech101'; % Update this path to your dataset location
outputFolder = 'path/to/output'; % Update this path to your desired output location
categories = {'airplanes', 'laptop', 'ferry'}; % Define the categories you're interested in

% Create an ImageDatastore
imds = imageDatastore(fullfile(rootFolder, categories), 'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Display the distribution of labels
tb1 = countEachLabel(imds);

% Split the dataset into training and testing sets
minSetCount = min(tb1);
[trainingSet, testSet] = splitEachLabel(imds, minSetCount, 'randomize');

% Display some images
figure;
subplot(2, 2, 1);
imshow(readImage(imds, trainingSet(1)));
title('Airplane');
subplot(2, 2, 2);
imshow(readImage(imds, trainingSet(2)));
title('Laptop');
subplot(2, 2, 3);
imshow(readImage(imds, trainingSet(3)));
title('Ferry');

% Load the ResNet-50 model
net = resnet50;

% Display the architecture of the model
figure;
plot(net);
title('Architecture of ResNet-50');

% Set the input size for the image data
imageSize = net.Layers(1).InputSize;

% Split the dataset into training and testing sets
[trainingSet, testSet] = splitEachLabel(imds, minSetCount, 'randomize');

% Augment the training and testing sets
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');

% Extract features from the augmented training set
featureLayer = net.Layers(end-1); % Assuming the last layer is the classification layer
trainingFeatures = activations(net, augmentedTrainingSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

% Train a classifier
trainingLabels = trainingSet.Labels;
classifier = fitcecoc(trainingFeatures, trainingLabels, 'Linear', 'Linear', 'Coding', 'onevsall', 'ObservationsIn', 'columns');

% Extract features from the augmented test set
testFeatures = activations(net, augmentedTestSet, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

% Predict labels for the test set
predictLabels = predict(classifier, testFeatures, 'ObservationsIn', 'columns');

% Compute the confusion matrix
confMat = confusionmat(testSet.Labels, predictLabels);
confMat = bsxfun(@rdivide, confMat, sum(confMat, 2));
meanAccuracy = mean(diag(confMat));

% Load a new image for prediction
newImage = imread('path/to/testImage.png'); % Update this path to your test image
augmentedImage = augmentedImageDatastore(imageSize, newImage, 'ColorPreprocessing', 'gray2rgb');
imageFeatures = activations(net, augmentedImage, featureLayer, 'MiniBatchSize', 32, 'OutputAs', 'columns');

% Predict the class of the new image
[~, predictedLabel] = predict(classifier, imageFeatures, 'ObservationsIn', 'columns');

% Display the prediction
sprintf('The loaded image belongs to class %s\n', predictedLabel);