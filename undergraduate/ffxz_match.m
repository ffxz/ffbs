

t1 = imread('t1.bmp');
t3 = imread('t3.bmp');
%origin_point = detectCheckerboardPoints('t1.bmp');
%dis_point = detectCheckerboardPoints('t3.bmp');

t1_undis = undistortImage(t1,params);
t3_undis = undistortImage(t3,params);
un_origin_point = detectCheckerboardPoints(t1_undis);%畸变矫正后的图像，检测角点
un_dis_point = detectCheckerboardPoints(t3_undis);

%[tform,inlierp,inorigin] = estimateGeometricTransform(dis_point,origin_point,'projective');
[tform,inlierp,inorigin] = estimateGeometricTransform(un_dis_point,un_origin_point,'projective');
figure;
%showMatchedFeatures(t1,t3,inorigin,inlierp);
showMatchedFeatures(t1_undis,t3_undis,inorigin,inlierp);

%outputView = imref2d(size(t1));
%Ir = imwarp(t3,tform,'outputView',outputView);
outputView = imref2d(size(t1_undis));
Ir = imwarp(t3_undis,tform,'outputView',outputView);
figure;
imshow(Ir);
title('recovered immage');
figure;
imshow(t1);



