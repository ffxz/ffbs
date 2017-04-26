%                        *
%                      *   *
%                   *  *   *  *


t4 = imread('t1.bmp');%t4Ϊֱ�������ͼ��
t4 = Ir;%���ڲ���ͼ����׼֮���ͼ��
J = undistortImage(t4,params);%JΪ����������ͼ��
imshow(J);
measure_point = detectCheckerboardPoints(J);%�ӽ�����ͼ������ȡ������
num = 1;%��¼u1��v1�ĸ���     ����
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

%����


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

angle_jiguang = (44.4-0.48)*2*pi/360;%input('�������ǽǶȣ�');%�������ǲ����ĽǶ�(�Ƕ�Ҫ����ɻ��ȼ���)-0.48
distance_jiguang = (0.648+0.027)*1e+03;%input('�������Ǿ��룺');%�������ǲ�õľ���(ע�����ĵ�λҪͳһ)+0.03
Nm = num-1;
H = distance_jiguang*sin(angle_jiguang);%����ĸ߳�
arfa = angle_jiguang;%������
thita = -(0-0)*2*pi/360;%input('����ǣ�');%�����
s = 0.0000052e+03;%��ԭ�ߴ�
fx = params.IntrinsicMatrix(1)*s;
fy = params.IntrinsicMatrix(5)*s;
f = (fx+fy)/2;%0.00826e+03;%input('����Ľ��ࣺ');%����Ľ���
x = params.IntrinsicMatrix(3)*2;%657.456*2;%1280-1;%input('�������ص���ܸ�����');%�������ص�(��Ϊ����ͼ���е�ʱ�ǰ���0~1279�����)
y = params.IntrinsicMatrix(6)*2;%479.2965*2;%1024-1;%504.9405*2;%%input('�������ص���ܸ�����');%�������ص�

thita_abs = abs(thita);

%ͼ����һ�����ص������
j1 = cos(thita)*((v1-y/2*ones(1,Nm)))-sin(thita)*(u1-x/2*ones(1,Nm))+y/2*ones(1,Nm);%(x*sin(thita_abs)+y*cos(thita_abs))/2;
i1 = sin(thita)*((v1-y/2*ones(1,Nm)))+cos(thita)*(u1-x/2*ones(1,Nm))+x/2*ones(1,Nm);%(x*cos(thita_abs)+y*sin(thita_abs))/2;%x/2;
%ͼ������һ�����ص������
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


%���dis��ÿ��С�ĸ��Ӿ���
dis12 = Dis12(1:Nm);
real_row = [30*ones(1,Nm/2),30*ones(1,Nm/2)];
%real = [real_line real_row];
err_real = (dis12-real_row)./real_row 

err_x = 1:Nm;
stem(err_x,err_real);
hold on;plot(err_x,0.01*ones(Nm,1),'r-');
hold on;plot(err_x,-0.01*ones(Nm,1),'r-');
hold on;plot(err_x,0.02*ones(Nm,1),'g-');
hold on;plot(err_x,-0.02*ones(Nm,1),'g-');

real_y = err_real(1:84);%����С����
real_x = err_real(85:end);%����С����
real_y = reshape(real_y,7,12)';
real_x = reshape(real_x,11,8)';
num_x = 1:11;
num_y = 1:7;
figure;
bar(real_y);
figure;
plot(num_y,real_y,'-o');
xlabel('����λ��');
ylabel('���');
figure;
bar(real_x);
figure;
plot(num_x,real_x,'-o');
xlabel('����λ��');
ylabel('���');


dis_hang = Dis12(1:84);
dis_hang = reshape(dis_hang,7,12);
dis_hang = sum(dis_hang)
dis_lie =  Dis12(85:end);
dis_lie = reshape(dis_lie,8,11);
dis_lie = dis_lie';
dis_lie = sum(dis_lie)

%��ʾ���ݲ�����ͼ�񣬴���Ӿ���
dis12 = [dis_hang(1) dis_hang(6) dis_hang(12) dis_lie(1) dis_lie(4) dis_lie(8)];
real_row = 330*ones(1,3);
real_line = 210*ones(1,3);
real = [real_line real_row];
err_real = (dis12-real)./real 
bar(err_real);

