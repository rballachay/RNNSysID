FO_model_C = ss(0.9,0.2,1,0,1,'InputDelay',5);

FO_test_model_S = ss(-0.1,0.2,1,0,1,'InputDelay',5);

%% create MPC controller object with sample time
mpc1 = mpc(FO_model_C, 1);
%% specify prediction horizon
mpc1.PredictionHorizon = 10;
%% specify control horizon
mpc1.ControlHorizon = 2;
%% specify nominal values for inputs and outputs
mpc1.Model.Nominal.U = 0;
mpc1.Model.Nominal.Y = 0;
%% specify weights
mpc1.Weights.MV = 0;
mpc1.Weights.MVRate = 0.1;
mpc1.Weights.OV = 1;
mpc1.Weights.ECR = 100000;
%% specify simulation options
options = mpcsimopt();
options.Model = FO_test_model_S;
options.RefLookAhead = 'off';
options.MDLookAhead = 'off';
options.Constraints = 'off';
options.OpenLoop = 'off';
mpc1_MDSignal = wgn(101,1,1);
mpc1_RefSignal = .1*randn(101,1);
%% run simulation
%sim(mpc1, 101, mpc1_RefSignal, mpc1_MDSignal, options);
x = sim(mpc1, 101, mpc1_RefSignal, options);


plot(linspace(1,101,101),fft(x));

