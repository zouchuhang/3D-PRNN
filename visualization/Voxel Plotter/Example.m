%Autho: Itzik Ben Shabat
%Date: 10.5.15
% This scrypt illustrates the use of VoxelPlotter function to visualize
% voxel data stored in a 3d matrix

clear all
close all
clc
%Generating sinthetic input
gridesize=16;
R=8;
VoxelMat=zeros(gridesize,gridesize,gridesize);
for i=1:gridesize
    for j=1:gridesize
        for k=1:gridesize
            if (i-gridesize/2)^2+(j-gridesize/2)^2+(k-gridesize/2)^2<R^2
                VoxelMat(i,j,k)=1;
            end
        end
    end
end
[vol_handle]=VoxelPlotter(VoxelMat,1); 
%visual effects (I recommend using the FigureRotator function from MATLAB
%Centeral
view(3);
daspect([1,1,1]);
set(gca,'xlim',[0 gridesize], 'ylim',[0 gridesize], 'zlim',[0 gridesize]);