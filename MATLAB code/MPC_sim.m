N = 0.9;
D = 0.2;
denominators = [1 -D];

FO_model_true = tf(N,denominators,1);

FO_model_python = tf(N+0.0140 ,[1 (-D-0.0107)],1);

FO_model_MATLAB = tf(N+0.0031,[1 (-D-0.0035)],1);

mpcDesigner();
