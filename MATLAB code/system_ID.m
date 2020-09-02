% Each row represents a different sample of the SISO system. In order to
% get data in .mat form, I'm going to open each CSV and make a .mat file
% for each input/output pair

csv_in = readtable('u_1x1.csv');
csv_out = readtable('y_1x1.csv');

for i = 1:1000
    u = csv_in{i,:};
    y = csv_out{i,:};
    name = 'SISO Data/'+string(i)+'.mat';
    save(name,'u','y');
end

taus = zeros(100,1);
thetas = zeros(100,1);
kps = zeros(100,1);

tic
for i = 1:1000
    filename = 'SISO Data/'+string(i)+'.mat';
    load(filename);
    u = reshape(u,[100,1]);
    y = reshape(y,[100,1]);
    Ts=1; % Time step is one second
    data = iddata(y,u);
    sysTF = procest(data,'P1D');
    taus(i) = sysTF.Tp1;
    thetas(i) = sysTF.Td;
    kps(i) = sysTF.Kp;
end
toc
