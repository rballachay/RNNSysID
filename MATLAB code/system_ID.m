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

taus = zeros(1000,1);
thetas = zeros(1000,1);
kps = zeros(1000,1);
noises = zeros(1000,1);
errors = zeros(1000,2);

uArr = zeros(300,1000);
yArr = zeros(300,1000);
for i = 1:1000
    filename = 'SISO Data/'+string(i)+'.mat';
    load(filename);
    uArr(:,i) = reshape(u,[300,1]);
    yArr(:,i) = reshape(y,[300,1]);
end
tic

for i = 1:1000
    y=yArr(:,i);
    u=uArr(:,i);
    Ts=1; % Time step is one second
    data = [y,u];
    sysTF = tfest(data,1,0,1,'Ts',1.0);
    kps(i) = sysTF.Numerator;
    taus(i) = sysTF.Denominator(2);
    thetas(i) = sysTF.IODelay;
    noises(i) = sysTF.NoiseVariance;
    temp = getcov(sysTF);
    errors(i,1) = temp(1,1);
    errors(i,2) = temp(2,2);
    
end
toc
