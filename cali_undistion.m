
%tic
t1 = clock;
N = 7;
images = cell(1,N);
imagesUsed =cell(1,N);
squareSize = 30;
%��ȡN�����̸�ͼ��ǵ�����
for ai =1:N
imagename = sprintf('0%d.bmp',ai);  
images{ai} = imread(imagename);
[imagePoints(:,:,ai),boardSize,imagesUsed{ai}] = detectCheckerboardPoints(images{ai});
end
%��ȡ���̸���������
[worldPoints] = generateCheckerboardPoints(boardSize,squareSize);
%��ȡ����궨�Ĳ���
params = estimateCameraParameters(imagePoints,worldPoints);
%�������ܣ���ȡ����ͼ��ǵ������

showReprojectionErrors(params);

t2 = clock;
etime(t2,t1)
%toc
