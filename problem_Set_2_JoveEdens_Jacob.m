data = csvread('CPI.csv',1,1);

cpi = data( : , 1 );        %cpi
hst =  data( : , 2 );       %Housing starts
tbill =  data( : , 3 );      %10 year t-bond yields
sp500 = data( : , 4 );      %sp500 total returns
pro = data( : , 5 );        %US industrial production growth finished goods 

X = [ hst tbill sp500 pro ]

logGrwthCpi = log(cpi( 2:end )./ cpi( 1:end-1 )); 
stat = [ 100*4*mean(logGrwthCpi) 100*sqrt(4)*std(logGrwthCpi) sqrt(4)*mean(logGrwthCpi)/std(logGrwthCpi) ]';

qtable = table(stat, ...
    'RowNames', { 'Mean (%)' , 'Vol (%)' , 'Sharpe' });
disp(qtable);

hstst = [ 100*4*mean(hst) 100*sqrt(4)*std(hst) sqrt(4)*mean(hst)/std(hst) ]';
tbillst = [ 100*4*mean(tbill) 100*sqrt(4)*std(tbill) sqrt(4)*mean(tbill)/std(tbill) ]';
sp500st = [ 100*4*mean(sp500) 100*sqrt(4)*std(sp500) sqrt(4)*mean(sp500)/std(sp500) ]';
prost = [ 100*4*mean(pro) 100*sqrt(4)*std(pro) sqrt(4)*mean(pro)/std(pro) ]';

disp(hstst)
disp(tbillst)
disp(sp500st)
disp(prost)

%%

y = logGrwthCpi;
T = length(y);
M1 = ((1969-1959)*12)-1; 
M2 = T-M1;
h = 1;

J = size(X, 2);

ForecastAR      = nan(M2-(h-1), 1);  % AR forecast
ForecastARDL    = nan(M2-(h-1), J);  % ARDL forecasts
ForecastCombine = nan(M2-(h-1), 1);  % combination forecast
ForecastDI      = nan(M2-(h-1), 1);  % diffusion index forecast

p_max = 6;  % max lag for AR & ARDL models

for m2 = 1:M2-(h-1)  % iterate over out-of-sample period

    y_m2 = y( 1:M1+(m2-1) );      % available observations
    X_m2 = X( 1:M1+(m2-1) , : );

    p_star_AR_m2 = SelectLagAR(y_m2, h, p_max);

    if p_star_AR_m2( 1 ) > 0

        alpha_hat_m2 = EstimateAR(y_m2, h, p_star_AR_m2( 1 ));

        y_m2_last = flipud(y_m2( end-(p_star_AR_m2( 1 )-1):end ));

        ForecastAR( m2 ) = [ 1 y_m2_last' ]*alpha_hat_m2;

    else

        ForecastAR( m2 ) = mean(y_m2);

    end

    for j = 1:J

        p_star_ARDL_m2_j = SelectLagARDL(y_m2, X_m2( : , j ), h, p_max);

        gamma_hat_m2_j = EstimateARDL(y_m2, X_m2( : , j ), h, ...
            p_star_ARDL_m2_j( : , 1 ));

        x_m2_j_last = flipud(...
            X_m2( end-(p_star_ARDL_m2_j( 2 , 1 )-1):end , j ));

        if p_star_ARDL_m2_j( 1 , 1 ) > 0

            y_m2_j_last = flipud(...
                y_m2( end-(p_star_ARDL_m2_j( 1 , 1 )-1):end ));

            ForecastARDL( m2 , j ) = ...
                [ 1 y_m2_j_last' x_m2_j_last' ]*gamma_hat_m2_j;

        else

            ForecastARDL( m2 , j ) = [ 1 x_m2_j_last' ]*gamma_hat_m2_j;

        end

    end

    ForecastCombine( m2 ) = mean(ForecastARDL( m2 , : ), 2);

    X_tilde_m2 = zscore(X_m2);

    [g_hat_m2, ~] = eigs(X_tilde_m2'*X_tilde_m2, 1);

    f_hat_m2 = X_tilde_m2*g_hat_m2;

    p_star_DI_m2 = SelectLagARDL(y_m2, f_hat_m2, h, p_max);

    gamma_hat_DI_m2 = EstimateARDL(y_m2, f_hat_m2, h, ...
        p_star_DI_m2( : , 1 ));

    f_hat_m2_last = flipud(f_hat_m2( end-(p_star_DI_m2( 2 , 1 )-1):end ));

    if p_star_DI_m2( 1 , 1 ) > 0

        y_m2_last = flipud(y_m2( end-(p_star_DI_m2( 1 , 1 )-1):end ));

        ForecastDI( m2 ) = [ 1 y_m2_last' f_hat_m2_last' ]*gamma_hat_DI_m2;

    else

        ForecastDI( m2 ) = [ 1 f_hat_m2_last' ]*gamma_hat_DI_m2;

    end

    disp([ m2 ForecastAR( m2 ) ForecastCombine( m2 ) ForecastDI( m2 ) ]);
    
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

u_hat_AR = Actual - ForecastAR;
MSFE_AR  = mean(u_hat_AR.^2);

u_hat_ARDL = repmat(Actual, 1, J) - ForecastARDL;
MSFE_ARDL  = mean(u_hat_ARDL.^2);

u_hat_Combine = Actual - ForecastCombine;
MSFE_Combine  = mean(u_hat_Combine.^2);

u_hat_DI = Actual - ForecastDI;
MSFE_DI  = mean(u_hat_DI.^2);

MSFE_ratio = [ MSFE_ARDL/MSFE_AR MSFE_Combine/MSFE_AR MSFE_DI/MSFE_AR ]';

%%
% Compute Clark-West statistics

d_hat_ARDL = repmat(u_hat_AR.^2, 1, J) - u_hat_ARDL.^2;
f_hat_ARDL = d_hat_ARDL + (repmat(ForecastAR, 1, J) - ForecastARDL).^2;

d_hat_Combine = u_hat_AR.^2 - u_hat_Combine.^2;
f_hat_Combine = d_hat_Combine + (ForecastAR - ForecastCombine).^2;

d_hat_DI = u_hat_AR.^2 - u_hat_DI.^2;
f_hat_DI = d_hat_DI + (ForecastAR - ForecastDI).^2;

CW = nan(J+2, 1);

if h == 1

    for j = 1:J

        Results_j = ols(f_hat_ARDL( : , j ), ...
            ones(length(f_hat_ARDL( : , j )), 1));

        CW( j ) = Results_j.tstat;

    end

    Results_Combine = ols(f_hat_Combine, ones(length(f_hat_Combine), 1));

    CW( J+1 ) = Results_Combine.tstat;

    Results_DI = ols(f_hat_DI, ones(length(f_hat_DI), 1));

    CW( J+2 ) = Results_DI.tstat;

else

    for j = 1:J

        Results_j = nwest(f_hat_ARDL( : , j ), ...
            ones(length(f_hat_ARDL( : , j )), 1), h-1);

        CW( j ) = Results_j.tstat;

    end

    Results_Combine = nwest(f_hat_Combine, ...
        ones(length(f_hat_Combine), 1), h-1);

    CW( J+1 ) = Results_Combine.tstat;

    Results_DI = nwest(f_hat_DI, ones(length(f_hat_DI), 1), h-1);

    CW( J+2 ) = Results_DI.tstat;

end

 MSFE_table = table(MSFE_ratio, CW, ...
    'RowNames', { 'TERM' , 'CREDIT', 'DY' , 'INFL' , 'Combine', 'DI' });

 disp(MSFE_table);

%%
% Graph forecasts

figure(1)

Month = ( 1970+0/12:1/12:2017+11/12 )';

Zeroline = zeros(length(Month), 1);

subplot(2, 2, 1);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastARDL( : , 1 ), '-b', 'LineWidth', 1.1);

    hold off

    title('ARDL forecast (%) Housing Starts', 'fontsize', 16);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 2);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastARDL( : , 2 ), '-b', 'LineWidth', 1.1);

    hold off

    title('ARDL forecast (%) based on T-Bill', 'fontsize', 16);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 3);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastARDL( : , 3 ), '-b', 'LineWidth', 1.1);

    hold off

    title('ARDL forecast (%) based on Sp500', 'fontsize', 16);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 2, 4);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastARDL( : , 4 ), '-b', 'LineWidth', 1.1);

    hold off

    title('ARDL forecast (%) based on US Production Final Goods', 'fontsize', 16);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 14);
    set(gca, 'ytick', -4:2:4, 'fontsize', 14);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

figure(2)

subplot(2, 1, 1);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastCombine, '-b', 'LineWidth', 1.1);

    hold off

    title('Combination forecast (%)', 'fontsize', 18);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 16);
    set(gca, 'ytick', -4:2:4, 'fontsize', 16);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);

subplot(2, 1, 2);

    plot(Month, Zeroline, '-k');

    hold on

    plot(Month, 100*Actual, '-k', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastAR, '-r', 'LineWidth', 1.1);

    hold on

    plot(Month, 100*ForecastDI, '-b', 'LineWidth', 1.1);

    hold off

    title('Diffusiuon index forecast (%)', 'fontsize', 18);

    set(gca, 'xtick', 1970:10:2010, 'fontsize', 16);
    set(gca, 'ytick', -4:2:4, 'fontsize', 16);
    xlabel('time');
    ylabel('inflation');

    axis([ Month( 1 ) Month( end ) -5 4 ]);











