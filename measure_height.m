%                        *
%                      *   *
%                   *  *   *  *


t4 = imread('s4.bmp');%t4为直接拍摄的图像
J = undistortImage(t4,params);%J为畸变矫正后的图像
imshow(J);
measure_point = 0;
[u1,v1] = ginput(1);
[u2,v2] = ginput(1);

angle_jiguang = ((90-7.9)-0)*2*pi/360;%input('激光测距仪角度：');%激光测距仪测量的角度(角度要换算成弧度计算)-0.48
distance_jiguang = (3.064-0.175)*1e+03;%input('激光测距仪距离：');%激光测距仪测得的距离(注意距离的单位要统一)+0.03
H = distance_jiguang*sin(angle_jiguang);%相机的高程
arfa = angle_jiguang;%俯仰角
thita = -(4.2-0)*2*pi/360;%input('横滚角：');%横滚角
s = 0.0000052e+03;%像原尺寸
fx = params.IntrinsicMatrix(1)*s;
fy = params.IntrinsicMatrix(5)*s;
f = (fx+fy)/2;%0.00826e+03;%input('相机的焦距：');%相机的焦距
x = params.IntrinsicMatrix(3)*2;%657.456*2;%1280-1;%input('横向像素点的总个数：');%横向像素点(因为计算图像中点时是按照0~1279计算的)
y = params.IntrinsicMatrix(6)*2;%479.2965*2;%1024-1;%504.9405*2;%%input('纵向像素点的总个数：');%纵向像素点

thita_abs = abs(thita);
Nm = 1;
%图像中一个像素点的坐标
j1 = cos(thita)*((v1-y/2*ones(1,Nm)))-sin(thita)*(u1-x/2*ones(1,Nm))+y/2*ones(1,Nm);%(x*sin(thita_abs)+y*cos(thita_abs))/2;
i1 = sin(thita)*((v1-y/2*ones(1,Nm)))+cos(thita)*(u1-x/2*ones(1,Nm))+x/2*ones(1,Nm);%(x*cos(thita_abs)+y*sin(thita_abs))/2;%x/2;
%图像中另一个像素点的坐标
j2 = cos(thita)*((v2-y/2*ones(1,Nm)))-sin(thita)*(u2-x/2*ones(1,Nm))+y/2*ones(1,Nm);%(x*sin(thita_abs)+y*cos(thita_abs))/2;%y/2;
i2 = sin(thita)*((v2-y/2*ones(1,Nm)))+cos(thita)*(u2-x/2*ones(1,Nm))+x/2*ones(1,Nm);%(x*cos(thita_abs)+y*sin(thita_abs))/2;%x/2;

H = H*ones(1,Nm);
angle_point1 = arfa + atan(-(y/2-j1)*s/f);
L1 = H./sin(angle_point1);
M1 = -(i1-x/2*ones(1,Nm))*s.*L1./sqrt(f^2*ones(1,Nm)+((y/2*ones(1,Nm)-j1)*s).^2);
N1 = H./tan(angle_point1);

angle_point2 = arfa + atan(-(y/2-j2)*s/f);
L2 = H./sin(angle_point2);
M2 = -(i2-x/2*ones(1,Nm))*s.*L2./sqrt(f^2*ones(1,Nm)+((y/2*ones(1,Nm)-j2)*s).^2);
N2 = H./tan(angle_point2);

Dis12 = sqrt((M2-M1).^2+(N2-N1).^2)


