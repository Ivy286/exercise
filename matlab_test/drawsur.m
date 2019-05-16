function drawsur= drawsur(surface)
%UNTITLED6 Summary of this function goes here
%   Detailed explanation goes here
    switch surfce
        case 'sphere'
            sphere
        case 'cylinder'
            cylinder
end
shading initerp
axis equal

drawsur('sphere')