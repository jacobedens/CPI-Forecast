clear
clc

%%
%prob 2 section 2

Data = csvread('WIL.csv',1,1);

STOCK    = Data( : , 1 );  % Wilshire 3000 return Monthly
BILL     = Data( : , 2 );  % GFD T-bill total return index
hst      = Data( : , 3 );  % housing starts
oil      = Data( : , 4 );  % Texas oil prices
div      = Data( : , 5 );  % SP500 dividend yeild
pro      = Data( : , 6 );  % US production (final goods)

Rm = STOCK( 2:end )./STOCK( 1:end-1 ) - 1;  % market return
Rf = BILL( 2:end )./BILL( 1:end-1 ) - 1;    % risk-free return
Rx =  Rm - Rf;
X  = [hst oil div pro]

stat = [ 100*12*mean(Rx) 100*sqrt(12)*std(Rx) sqrt(12)*mean(Rx)/std(Rx) ]';
xstat = [ 100*12*mean(X) 100*sqrt(12)*std(X) sqrt(12)*mean(X)/std(X) ]';

Rx_stat_ann_table = table(stat, ...
    'RowNames', { 'Mean (%)' , 'Vol (%)' , 'Sharpe' });

disp('Annualized summary statistics - market excess return');
disp(Rx_stat_ann_table);

hstst = [ 100*4*mean(hst) 100*sqrt(4)*std(hst) sqrt(4)*mean(hst)/std(hst) ]';
oilst = [ 100*4*mean(oil) 100*sqrt(4)*std(oil) sqrt(4)*mean(oil)/std(oil) ]';
divst = [ 100*4*mean(div) 100*sqrt(4)*std(div) sqrt(4)*mean(div)/std(div) ]';
prost = [ 100*4*mean(pro) 100*sqrt(4)*std(pro) sqrt(4)*mean(pro)/std(pro) ]';

disp(hstst)
disp(oilst)
disp(divst)
disp(prost)
%%
y  = Rx;                % target variable
T  = length(y);         % total number of observations
M1 = (1988 - 1978)*12;  % 1978:01-1988:12 in-sample period
M2 = T - M1;            % 1989:01-2017:12 out-of-sample period
h = 1;                  % forecast horizon

J = size(X, 2);

ForecastPM      = nan(M2-(h-1), 1);  % prevailing mean forecast
ForecastPR      = nan(M2-(h-1), J);  % predictive regression forecasts
ForecastCombine = nan(M2-(h-1), 1);  % combination forecast
ForecastDI      = nan(M2-(h-1), 1);  % diffusion index forecast

for m2 = 1:M2-(h-1)  % iterate over out-of-sample period

    y_m2 = y( 1:M1+(m2-1) );      % available observations
    X_m2 = X( 1:M1+(m2-1) , : );

    ForecastPM( m2 ) = mean(y_m2);

    for j = 1:J

        gamma_hat_m2_j = EstimateARDL(y_m2, X_m2( : , j ), h, [ 0 1 ]);

        ForecastPR( m2 , j ) = [ 1 X_m2( end , j ) ]*gamma_hat_m2_j;

    end

    ForecastCombine( m2 ) = mean(ForecastPR( m2 , : ), 2);

    X_tilde_m2 = zscore(X_m2); %zscore fn = column based standardization ((x_jt - x_samp mean)/std) 

    [g_hat_m2, ~] = eigs(X_tilde_m2'*X_tilde_m2, 1); %digs computes eigvect and eigval; we only need 1st. ~ says, do not store( in our case, do not save eigen value).

    f_hat_m2 = X_tilde_m2*g_hat_m2;

    gamma_hat_m2 = EstimateARDL(y_m2, f_hat_m2, h, [ 0 1 ]);

    ForecastDI(m2) = [ 1 f_hat_m2(end) ]*gamma_hat_m2;

    disp([ m2 ForecastPM( m2 ) ForecastCombine( m2 ) ForecastDI( m2 ) ]);

end

%%
% Compute actual values over out-of-sample period

if h > 1

    y_h = zeros(T-(h-1), 1);

    for t = 1:T-(h-1)

        y_h( t ) = mean(y( t:t+(h-1) ));

    end

else

    y_h = y;

end

Actual = y_h( M1+1:end );

%%
% Compute mean squared forecast errors

u_hat_PM = Actual - ForecastPM;
MSFE_PM  = mean(u_hat_PM.^2);

u_hat_PR = repmat(Actual, 1, J) - ForecastPR;
MSFE_PR  = mean(u_hat_PR.^2);

u_hat_Combine = Actual - ForecastCombine;
MSFE_Combine  = mean(u_hat_Combine.^2);

u_hat_DI = Actual - ForecastDI;
MSFE_DI  = mean(u_hat_DI.^2);

MSFE_ratio = [ MSFE_PR/MSFE_PM MSFE_Combine/MSFE_PM MSFE_DI/MSFE_PM ]';

%%
% Compute Clark-West statistics

d_hat_PR = repmat(u_hat_PM.^2, 1, J) - u_hat_PR.^2;
f_hat_PR = d_hat_PR + (repmat(ForecastPM, 1, J) - ForecastPR).^2;

d_hat_Combine = u_hat_PM.^2 - u_hat_Combine.^2;
f_hat_Combine = d_hat_Combine + (ForecastPM - ForecastCombine).^2;

d_hat_DI = u_hat_PM.^2 - u_hat_DI.^2;
f_hat_DI = d_hat_DI + (ForecastPM - ForecastDI).^2;

CW = nan(J+2, 1);

if h == 1

    for j = 1:J

        Results_j = ols(f_hat_PR( : , j ), ...
            ones(length(f_hat_PR( : , j )), 1));

        CW( j ) = Results_j.tstat;

    end

    Results_Combine = ols(f_hat_Combine, ones(length(f_hat_Combine), 1));

    CW( J+1 ) = Results_Combine.tstat;

    Results_DI = ols(f_hat_DI, ones(length(f_hat_DI), 1));

    CW( J+2 ) = Results_DI.tstat;

else

    for j = 1:J

        Results_j = nwest(f_hat_PR( : , j ), ...
            ones(length(f_hat_PR( : , j )), 1), h-1);

        CW( j ) = Results_j.tstat;

    end

    Results_Combine = nwest(f_hat_Combine, ...
        ones(length(f_hat_Combine), 1), h-1);

    CW( J+1 ) = Results_Combine.tstat;

    Results_DI = nwest(f_hat_DI, ones(length(f_hat_DI), 1), h-1);

    CW( J+2 ) = Results_DI.tstat;

end

MSFE_table = table(MSFE_ratio, CW, ...
    'RowNames', { 'cpi' , 'busc', 'unem' , 'indp' , 'Combine', 'DI' });

disp(MSFE_table);

%%
figure(1)

Month = ( 1989+0/12:1/12:2017+11/12 )';

Zeroline = zeros(length(Month), 1);

subplot(2, 2, 1);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPR( : , 1 ), '-b', 'LineWidth', 1.1);

    hold off

    title('Predictive Regr forecast (%) based on hst', 'fontsize', 16);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
    xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 2);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPR( : , 2 ), '-b', 'LineWidth', 1.1);

    hold off

    title('Predictive Regr forecast (%) based on oil price', 'fontsize', 16);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
     xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 3);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPR( : , 3 ), '-b', 'LineWidth', 1.1);

    hold off

    title('PR forecast (%) based on SP500 dividend yield', 'fontsize', 16);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
     xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 4);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPR( : , 4 ), '-b', 'LineWidth', 1.1);

    hold off

    title('Predictive Regr forecast (%) based on production', 'fontsize', 16);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
     xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

figure(2)

subplot(2, 1, 1);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastCombine, '-b', 'LineWidth', 1.1);

    hold off

    title('Combination forecast (%)', 'fontsize', 18);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 16);
    set(gca, 'ytick', -4:2:4, 'fontsize', 16);
     xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 1, 2);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastPM, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastDI, '-b', 'LineWidth', 1.1);

    hold off

    title('Diffusiuon index forecast (%)', 'fontsize', 18);

    set(gca, 'xtick', 1960:10:2010, 'fontsize', 16);
    set(gca, 'ytick', -4:2:4, 'fontsize', 16);
     xlabel('time');
    ylabel('return');

    axis([ Month( 1 ) Month( end ) -5 4 ]);
 


 