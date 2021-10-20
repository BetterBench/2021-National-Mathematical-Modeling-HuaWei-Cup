%% 计算AQI主函数
clc,
clear all;

% data = xlsread("D:\matlab-work\数模国赛\data_1_linear(3).xlsx",'监测点A逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_1_linear(3).xlsx",'监测点A逐小时污染物浓度与气象实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_1_knn(3).xlsx",'监测点A逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_1_knn(3).xlsx",'监测点A逐小时污染物浓度与气象实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点B逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点B逐小时污染物浓度与气象实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点C逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点C逐小时污染物浓度与气象实测数据');

% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A1逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A1逐小时污染物浓度与气象实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A2逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A2逐小时污染物浓度与气象实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A3逐日污染物浓度实测数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A3逐小时污染物浓度与气象实测数据');
%  data = xlsread("D:\matlab-work\数模国赛\data_2_knn(2).xlsx",'监测点C逐日污染物浓度实测数据');
 data = xlsread("C:\Users\Lenovo\Desktop\Q2_实验结果_平均.xlsx",'Sheet2');
[row,column]=size(data);
so_2=data(:,1);
no_2=data(:,2);
pm_10=data(:,3);
pm_2_5=data(:,4);
o_3=data(:,5);
co=data(:,6);


% data = xlsread("D:\matlab-work\数模国赛\data_1_knn(3).xlsx",'监测点A逐小时污染物浓度与气象一次预报数据');
% data = xlsread("D:\matlab-work\数模国赛\data_1_linear(3).xlsx",'监测点A逐小时污染物浓度与气象一次预报数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点B逐小时污染物浓度与气象一次预报数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn.xlsx",'监测点C逐小时污染物浓度与气象一次预报数据');

% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A1逐小时污染物浓度与气象一次预报数据');
% data = xlsread("D:\matlab-work\数模国赛\data_2_knn(2).xlsx",'监测点C逐小时污染物浓度与气象一次预报数据');
% data = xlsread("D:\matlab-work\数模国赛\data_3_knn.xlsx",'监测点A3逐小时污染物浓度与气象一次预报数据');

% [row,column]=size(data);
% so_2=data(:,16);
% no_2=data(:,17);
% pm_10=data(:,18);
% pm_2_5=data(:,19);
% o_3=data(:,20);
% co=data(:,21);


% so_2=12;
% no_2=66;
% pm_10=83;
% pm_2_5=39;
% o_3=210;
% co=0.8;
% row=1;
% so_2=20;no_2=45;pm_10=75;pm_2_5=45;o_3=300;co=0.7;row=1;
% IAQI_so_2=IAQI(so_2,"IAQI_SO_2");
% IAQI_no_2=IAQI(no_2,"IAQI_NO_2");
% IAQI_pm_10=IAQI(pm_10,"IAQI_PM_10");
% IAQI_pm_25=IAQI(pm_2_5,"IAQI_PM_25");
% IAQI_o_3=IAQI(o_3,"IAQI_O_3");
% IAQI_co=IAQI(co,"IAQI_CO");
for i=1:row
    IAQI_so_2(i)=IAQI(so_2(i),"IAQI_SO_2");
    IAQI_no_2(i)=IAQI(no_2(i),"IAQI_NO_2");
    IAQI_pm_10(i)=IAQI(pm_10(i),"IAQI_PM_10");
    IAQI_pm_25(i)=IAQI(pm_2_5(i),"IAQI_PM_25");
    IAQI_o_3(i)=IAQI(o_3(i),"IAQI_O_3");
    IAQI_co(i)=IAQI(co(i),"IAQI_CO");
end

IAQI=[IAQI_so_2;IAQI_no_2;IAQI_pm_10;IAQI_pm_25;IAQI_o_3;IAQI_co];
[IAQ,I]=max(IAQI);
FP=FP(IAQ,I,row);
for i=1:5
    for j=I(i)+1:6
        if IAQI(j,i)==IAQ(i)&&I(i)~=i&&IAQ(i)>50
            FP(i)=FP(i)+"、"+FPp(j);
        end
    end
end
A=FP';
B=IAQ';
C=IAQI';
% xlswrite("D:\matlab-work\数模国赛\data_3_knn.xlsx",B,'监测点A2逐小时污染物浓度与气象一次预报数据','W2')
% xlswrite("D:\matlab-work\数模国赛\data_3_knn.xlsx",A,'监测点A2逐小时污染物浓度与气象一次预报数据','X2')
% xlswrite("D:\matlab-work\数模国赛\data_3_knn.xlsx",C,'监测点A2逐小时污染物浓度与气象一次预报数据','Y2')

%%IAQI.m文件是用来计算各项污染物 的IAQI指数

%%
function IAQI=IAQI(c,problem)
if problem=="IAQI_CO"
    IAQI=IAQI_CO(c);
elseif problem=="IAQI_SO_2"
    IAQI=IAQI_SO_2(c);
elseif problem=="IAQI_NO_2"
    IAQI=IAQI_NO_2(c);
elseif problem=="IAQI_O_3"
    IAQI=IAQI_O_3(c);
elseif problem=="IAQI_PM_10"
    IAQI=IAQI_PM_10(c);
elseif problem=="IAQI_PM_25"
    IAQI=IAQI_PM_25(c);
else 
    print("输入错误");
end

function IAQI=IAQI_CO(c)
if c>=0&&c<=2
    IAQI=ceil(50/(2-0)*(c-0)+0);
elseif c>2&&c<=4
    IAQI=ceil(50/(4-2)*(c-2)+50);
elseif c>4&&c<=14
    IAQI=ceil(50/(14-4)*(c-4)+100);
elseif c>14&&c<=24
    IAQI=ceil(50/(24-14)*(c-14)+150);
elseif c>24&&c<=36
    IAQI=ceil(50/(36-24)*(c-24)+200);
elseif c>36&&c<=48
    IAQI=ceil(50/(48-36)*(c-36)+300);
elseif c>48&&c<=60
    IAQI=ceil(50/(48-36)*(c-36)+400);
elseif c>60
    IAQI=NaN;   
else
    print("输入错误");
end

function IAQI=IAQI_SO_2(c)
if c>=0&&c<=50
    IAQI=ceil(50/(50-0)*(c-0)+0);
elseif c>50&&c<=150
    IAQI=ceil(50/(150-50)*(c-50)+50);
elseif c>150&&c<=475
    IAQI=ceil(50/(475-150)*(c-150)+100);
elseif c>475&&c<=800
    IAQI=ceil(50/(800-475)*(c-475)+150);
elseif c>800&&c<=1600
    IAQI=ceil(50/(1600-800)*(c-800)+200);
elseif c>1600&&c<=2100
    IAQI=ceil(50/(2100-1600)*(c-1600)+300);
elseif c>2100&&c<=2620
    IAQI=ceil(50/(2620-2100)*(c-2100)+400);
elseif c>2620
    IAQI=NaN;
else 
    print("输入错误");      
end

function IAQI=IAQI_NO_2(c)
if c>=0&&c<=40
    IAQI=ceil(50/(40-0)*(c-0)+0);
elseif c>40&&c<=80
    IAQI=ceil(50/(80-40)*(c-40)+50);
elseif c>80&&c<=180
    IAQI=ceil(50/(180-80)*(c-80)+100);
elseif c>180&&c<=280
    IAQI=ceil(50/(280-180)*(c-180)+150);
elseif c>280&&c<=565
    IAQI=ceil(50/(565-280)*(c-280)+200);
elseif c>565&&c<=750
    IAQI=ceil(50/(750-565)*(c-565)+300);
elseif c>750&&c<=940
    IAQI=ceil(50/(940-750)*(c-750)+400);
elseif c>2620
    IAQI=NaN;
else 
    print("输入错误");      
end

function IAQI=IAQI_O_3(c)
if c>=0&&c<=100
    IAQI=ceil(50/(100-0)*(c-0)+0);
elseif c>100&&c<=160
    IAQI=ceil(50/(160-100)*(c-100)+50);
elseif c>160&&c<=215
    IAQI=ceil(50/(215-160)*(c-160)+100);
elseif c>215&&c<=265
    IAQI=ceil(50/(265-215)*(c-215)+150);
elseif c>265&&c<=800
    IAQI=ceil(50/(800-265)*(c-265)+200);
elseif c>800
    IAQI=NaN;
else 
    print("输入错误");      
end

function IAQI=IAQI_PM_10(c)
if c>=0&&c<=50
    IAQI=ceil(50/(50-0)*(c-0)+0);
elseif c>50&&c<=150
    IAQI=ceil(50/(150-50)*(c-50)+50);
elseif c>150&&c<=250
    IAQI=ceil(50/(250-150)*(c-150)+100);
elseif c>250&&c<=350
    IAQI=ceil(50/(350-250)*(c-250)+150);
elseif c>350&&c<=420
    IAQI=ceil(50/(420-350)*(c-350)+200);
elseif c>420&&c<=500
    IAQI=ceil(50/(500-420)*(c-420)+300);
elseif c>500&&c<=600
    IAQI=ceil(50/(600-500)*(c-500)+400);
elseif c>600
    IAQI=NaN;
else 
    print("输入错误");      
end

function IAQI=IAQI_PM_25(c)
if c>=0&&c<=35
    IAQI=ceil(50/(35-0)*(c-0)+0);
elseif c>35&&c<=75
    IAQI=ceil(50/(75-35)*(c-35)+50);
elseif c>75&&c<=115
    IAQI=ceil(50/(115-75)*(c-75)+100);
elseif c>115&&c<=150
    IAQI=ceil(50/(150-115)*(c-115)+150);
elseif c>150&&c<=250
    IAQI=ceil(50/(250-150)*(c-150)+200);
elseif c>250&&c<=350
    IAQI=ceil(50/(3500-250)*(c-250)+300);
elseif c>350&&c<=500
    IAQI=ceil(50/(500-350)*(c-350)+400);
elseif c>500
    IAQI=NaN;
else 
    print("输入错误");      
end
