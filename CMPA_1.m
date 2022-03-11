Is = 0.01E-12;
Ib = 0.1E-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);

I = Is*(exp((1.2/0.025).*V) - 1) + Gp.*V - Ib*(exp((-1.2/0.025).*(V + Vb)) - 1);


sd = 0.2;
noise = sd*I.*(rand(size(I))-0.5)*2;
Inoise = I + noise;


P4 = polyfit(V,Inoise,4);
P8 = polyfit(V,Inoise,8);

Y4 = polyval(P4,V);
Y8 = polyval(P8,V);


figure(1)
plot(V,I);
xlabel('Voltage (V)')
ylabel('Current (A)')
hold on

plot(V,Inoise);
hold on

plot(V,Y4);
hold on

plot(V,Y8);

legend('I','Inoise','poly4','poly8')
hold off


figure(2)
semilogy(V,abs(I));
xlabel('Voltage (V)')
ylabel('Abs Current (A)')
hold on

semilogy(V,abs(Inoise));
hold on

semilogy(V,abs(Y4));

semilogy(V,abs(Y8));

legend('I','Inoise','poly4','poly8')
hold off



%3a Conditions
B2 = Gp;
D2 = Vb;

fo2 = fittype('A2.*(exp(1.2*x/25e-3)-1) + B2.*x - C2*(exp(1.2*(-(x+D2))/25e-3)-1)');
ff2 = fit(V',Inoise',fo2);
If2 = ff2(V);

%3b Conditions
D3 = Vb;

fo3 = fittype('A3.*(exp(1.2*x/25e-3)-1) + B3.*x - C3*(exp(1.2*(-(x+D3))/25e-3)-1)');
ff3 = fit(V',Inoise',fo3);
If3 = ff3(V);

%3c Conditions

fo4 = fittype('A4.*(exp(1.2*x/25e-3)-1) + B4.*x - C4*(exp(1.2*(-(x+D4))/25e-3)-1)');
ff4 = fit(V',Inoise',fo4);
If4 = ff4(V);


figure(3)
plot(V,Inoise);
xlabel('Voltage (V)')
ylabel('Current (A)')
hold on

plot(V,If2);
hold on

plot(V,If3);
hold on

plot(V,If4);

legend('Inoise','fit2','fit3','fit4')
hold off


figure(4)
semilogy(V,abs(Inoise));
xlabel('Voltage (V)')
ylabel('Abs Current (A)')
hold on

semilogy(V,abs(If2));
hold on

semilogy(V,abs(If3));

semilogy(V,abs(If4));

legend('Inoise','fit2','fit3','fit4')
hold off



inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net)
Inn = outputs;


figure(5)
plot(V,Inoise);
xlabel('Voltage (V)')
ylabel('Current (A)')
hold on

plot(V,Inn);
hold on

legend('Inoise','NN')
hold off


figure(6)
semilogy(V,abs(Inoise));
xlabel('Voltage (V)')
ylabel('Abs Current (A)')
hold on

semilogy(V,abs(Inn));
hold on

legend('Inoise','NN')
hold off

