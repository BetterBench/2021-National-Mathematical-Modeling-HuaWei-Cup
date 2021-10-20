
clc
clear

load('./IAQI_data_3f/X_3all.mat')
load('./IAQI_data_3f/index1.mat')
load('./IAQI_data_3f/index2.mat')
load('./IAQI_data_3f/index3.mat')

% 时间与风速的分类散点图
X = X_all
K = 50
% 选择不同的col列，去绘制不同的特征的关系散点图
col = 7
Y1 = smooth(X_all(index1+1,col),K,'sgolay')
Y2 = smooth(X_all(index2+1,col),K,'sgolay')
Y3 = smooth(X_all(index3+1,col),K,'sgolay')
scatter(index1,Y1,'g','filled')

hold on
scatter(index2,Y2,'r','filled')
hold on
scatter(index3,Y3,'b','filled')

xlabel('小时')
ylabel('温度')
box on
title('三种分类-散点图');
legend('第一类','第二类','第三类');
