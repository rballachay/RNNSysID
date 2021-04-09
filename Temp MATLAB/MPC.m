
CSTR_internal = ss([-0.5 0.04 ; 0.04 0.05],[-0.05 .6 ; .5 0.05],[1 0;0 1],0,1,'InputDelay',1);
CSTR_external = ss([0.7 0.03 ; 0.2 0.6],[-0.04 .5 ; -0.09 0.06],[1 0;0 1],0,1,'InputDelay',5);


mpcDesigner