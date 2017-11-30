function [ vx ] = getVX( Rv )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

vx = [0, -Rv(3), Rv(2);
        Rv(3), 0, -Rv(1);
        -Rv(2), Rv(1), 0];
end

