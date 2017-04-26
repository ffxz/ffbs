
%tic
t1 = clock;
N = 12;
images = cell(1,N);
imagesUsed =cell(1,N);
squareSize = 30;
deltN = 0;%�޳��ڽǵ������ͼ��
%��ȡN�����̸�ͼ��ǵ�����
for ai =1:N
imagename = sprintf('0%d.bmp',ai);  
images{ai} = imread(imagename);
if(size(detectCheckerboardPoints(images{ai}),1)==96)
    [imagePoints(:,:,ai-deltN),boardSize,imagesUsed{ai-deltN}] = detectCheckerboardPoints(images{ai});
else
    deltN = deltN+1;
    disp(ai)
    disp('ͼ������')
end
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
