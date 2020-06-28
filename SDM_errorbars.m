function [ebs] = SDM_errorbars(x_pos,mean,err,color,lineWidth,tickWidth)

%
% INPUT: (x_pos,mean,err,color,lineWidth,tickWidth)
%
%
% this function adds error bars with horizontal ticks
% input is: x-position on graph, the data point (e.g. mean), size of SEM or CI,
% the color of the error bars (rgb vector or string), the width of the 
% lines, and the span of the horizontal ticks

ebs = ...
plot([x_pos x_pos],[mean-err mean+err],'Color',color,'LineWidth',lineWidth);
hold on;
plot([x_pos-tickWidth x_pos+tickWidth],[mean+err mean+err],'Color',color,'LineWidth',lineWidth);
plot([x_pos-tickWidth x_pos+tickWidth],[mean-err mean-err],'Color',color,'LineWidth',lineWidth);

