%                        *
%                      *   *
%                   *  *   *  *

t4 = imread('t1.bmp');%t4为直接拍摄的图像
%t4 = Ir;
J = undistortImage(t4,params);%J为畸变矫正后的图像
imshow(J);
measure_point = detectCheckerboardPoints(J);%从矫正后图像中提取测量点
num = 1;%记录u1，v1的个数     纵向
for mu1 = 1:12
    for mv1 = (mu1-1)*8+1:8*mu1-1
        u1(num) = measure_point(mv1,1);
        v1(num) = measure_point(mv1,2);
        num = num+1;
    end
end
num = 1;
for mu1 = 1:12
    for mv1 = (mu1-1)*8+2:8*mu1
        u2(num) = measure_point(mv1,1);
        v2(num) = measure_point(mv1,2);
        num = num+1;
    end
end

%横向


 for mu1 = 1:8
     for mv1 = 1:11
        u1(num) = measure_point(8*(mv1-1)+mu1,1);
        v1(num) = measure_point(8*(mv1-1)+mu1,2);
        num = num+1;
    end
end

num = 85;


for mu1 = 1:8
    for mv1 = 2:12
        u2(num) = measure_point(8*(mv1-1)+mu1,1);
        v2(num) = measure_point(8*(mv1-1)+mu1,2);
        num = num+1;
    end
end

angle_jiguang = (44.4-0.48)*2*pi/360;%input('激光测距仪角度：');%激光测距仪测量的角度(角度要换算成弧度计算)-0.48
distance_jiguang = (0.648+0.027)*1e+03;%input('激光测距仪距离：');%激光测距仪测得的距离(注意距离的单位要统一)+0.03
Nm = num-1;
H = distance_jiguang*sin(angle_jiguang);%相机的高程
%arfa = input('光轴的俯仰角度：');%俯仰角
arfa = angle_jiguang;
thita = -(0-0)*2*pi/360;%input('横滚角：');%横滚角
s = 0.0000052e+03;%像原尺寸
fx = params.IntrinsicMatrix(1)*s;
fy = params.IntrinsicMatrix(5)*s;
f = (fx+fy)/2;%0.00826e+03;%input('相机的焦距：');%相机的焦距
x = params.IntrinsicMatrix(3)*2;%657.456*2;%1280-1;%input('横向像素点的总个数：');%横向像素点(因为计算图像中点时是按照0~1279计算的)
y = params.IntrinsicMatrix(6)*2;%479.2965*2;%1024-1;%504.9405*2;%%input('纵向像素点的总个数：');%纵向像素点

thita_abs = abs(thita);

%图像中一个像素点的坐标
% u1 = [measure_point(1,1),measure_point(5,1),measure_point(8,1),measure_point(1,1),measure_point(49,1),measure_point(89,1)];
 %v1 = [measure_point(1,2),measure_point(5,2),measure_point(8,2),measure_point(1,2),measure_point(49,2),measure_point(89,2)];
% u1 = [155.832 572.316 898.465 155.832 161.75 169.509 151.74 572.11 899.515 151.74 157.616 164.681];%517.742;%input('第一个点横坐标');
% v1 = [301.955 235.988 183.157 301.955 456.648 641.596 301.546 235.754 182.246 301.546 457.38 644.138];%260.368;%input('第一个点纵坐标');
j1 = cos(thita)*((v1-y/2*ones(1,Nm)))-sin(thita)*(u1-x/2*ones(1,Nm))+y/2*ones(1,Nm);%(x*sin(thita_abs)+y*cos(thita_abs))/2;
i1 = sin(thita)*((v1-y/2*ones(1,Nm)))+cos(thita)*(u1-x/2*ones(1,Nm))+x/2*ones(1,Nm);%(x*cos(thita_abs)+y*sin(thita_abs))/2;%x/2;

%图像中另一个像素点的坐标
% u2 = [measure_point(89,1),measure_point(93,1),measure_point(96,1),measure_point(8,1),measure_point(56,1),measure_point(96,1)];
 %v2 = [measure_point(89,2),measure_point(93,2),measure_point(96,2),measure_point(8,2),measure_point(56,2),measure_point(96,2)];
% u2 = [169.509 666.641 1048.53 898.465 968.248 1048.53 164.681 666.622 1051.3 899.515 969.615 1051.3];%703.638;% input('第二个点横坐标');
% v2 = [641.596 548.14 476.406 183.157 317.18 476.406 644.138 548.205 476.797 182.246 316.908 476.797];%536.816;%input('第二个点纵坐标');
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



dis12 = Dis12(1:Nm);
real_row = [30*ones(1,Nm/2),30*ones(1,Nm/2)];
%real = [real_line real_row];
err_real = (dis12-real_row)./real_row 
% 
err_x = 1:Nm;
stem(err_x,err_real);
hold on;plot(err_x,0.01*ones(Nm,1),'r-');
hold on;plot(err_x,-0.01*ones(Nm,1),'r-');
hold on;plot(err_x,0.02*ones(Nm,1),'g-');
hold on;plot(err_x,-0.02*ones(Nm,1),'g-');

real_y = err_real(1:84);%纵向小格子
real_x = err_real(85:end);%横向小格子
real_y = reshape(real_y,7,12)';
real_x = reshape(real_x,11,8)';
num_x = 1:11;
num_y = 1:7;
figure;
bar(real_y);
figure;
plot(num_y,real_y,'-o');
xlabel('纵向位置');
ylabel('误差');
figure;
bar(real_x);
figure;
plot(num_x,real_x,'-o');
xlabel('横向位置');
ylabel('误差');


dis_hang = Dis12(1:84);
dis_hang = reshape(dis_hang,7,12);
sum(dis_hang)
dis_lie =  Dis12(85:end);
dis_lie = reshape(dis_lie,8,11);
dis_lie = dis_lie';
sum(dis_lie)


%无用代码
% for num_y = 1:12
% figure;
% plot([1:7],real_y(num_y,:),'-*');
% xlabel('纵向位置');
% end


