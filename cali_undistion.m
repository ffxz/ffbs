
%tic
t1 = clock;
N = 7;
images = cell(1,N);
imagesUsed =cell(1,N);
squareSize = 30;
%获取N幅棋盘格图像角点坐标
for ai =1:N
imagename = sprintf('0%d.bmp',ai);  
images{ai} = imread(imagename);
[imagePoints(:,:,ai),boardSize,imagesUsed{ai}] = detectCheckerboardPoints(images{ai});
end
%获取棋盘格世界坐标
[worldPoints] = generateCheckerboardPoints(boardSize,squareSize);
%获取相机标定的参数
params = estimateCameraParameters(imagePoints,worldPoints);
%后续功能，提取矫正图像角点的坐标

showReprojectionErrors(params);

t2 = clock;
etime(t2,t1)
%toc
