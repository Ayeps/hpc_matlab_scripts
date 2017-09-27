%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Exploiting Spatial Reuse in Wireless Networks through Decentralised MABs
% F. Wilhelmi, B. Bellalta, A. Jonsson, C. Cano, G. Neu, S. Barrachina
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [] = display_results_individual_performance( wlans, tptEvolutionPerWlan, ...
    timesArmHasBeenPlayed, method_name )

    load('constants.mat')

    % Set font type
    set(0,'defaultUicontrolFontName','Times New Roman');
    set(0,'defaultUitableFontName','Times New Roman');
    set(0,'defaultAxesFontName','Times New Roman');
    set(0,'defaultTextFontName','Times New Roman');
    set(0,'defaultUipanelFontName','Times New Roman');

    % Compute data to be plotted
    mean_agg_tpt = mean(sum(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations, :),2));
    mean_std_agg_tpt = mean(std(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations, :)));
    mean_fairness = mean(JainsFairness(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations, :)));
    mean_prop_fairness = mean(sum(log(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations, :)),2));
    
    disp(['Mean aggregate throughput in ' method_name ': ' num2str(mean_agg_tpt)])
    disp(['Mean Std agg. tpt in ' method_name ': ' num2str(mean_std_agg_tpt)])
    disp(['Mean fairness in ' method_name ': ' num2str(mean_fairness)])
    disp(['Mean Prop. fairness in ' method_name ': ' num2str(mean_prop_fairness)])

    n_wlans = size(wlans, 2);

    %% Aggregated throughput experienced for each iteration
    figure('pos',[450 400 500 350]);
    axes;
    axis([1 20 30 70]);
    agg_tpt_per_iteration = sum(tptEvolutionPerWlan(1:totalIterations, :), 2);
    plot(1:totalIterations, agg_tpt_per_iteration)
    set(gca,'FontSize', 22)
    xlabel([method_name ' iteration'], 'fontsize', 24)
    ylabel('Network Throughput (Mbps)', 'fontsize', 24)
    axis([1 totalIterations 0 1.1 * max(agg_tpt_per_iteration)])
    % Save Figure
    fig_name = ['temporal_aggregate_tpt_' method_name];
    savefig(['./Output/' fig_name '.fig'])
    saveas(gcf,['./Output/' fig_name],'epsc')
    
    %% Throughput experienced by each WLAN for each iteration
    fig = figure('pos',[450 400 500 350]);
    axes;
    axis([1 20 30 70]);
    for i = 1:n_wlans
        subplot(n_wlans/2, n_wlans/2, i)
        tpt_per_iteration = tptEvolutionPerWlan(1:totalIterations, i);
        plot(1:totalIterations, tpt_per_iteration);
        title(['WN ' num2str(i)]);
        set(gca, 'FontSize', 18)
        axis([1 totalIterations 0 1.1 * max(tpt_per_iteration)])
    end
    % Set Axes labels
    AxesH    = findobj(fig, 'Type', 'Axes');       
    % Y-label
    YLabelHC = get(AxesH, 'YLabel');
    YLabelH  = [YLabelHC{:}];
    set(YLabelH, 'String', 'Throughput (Mbps)')
    % X-label
    XLabelHC = get(AxesH, 'XLabel');
    XLabelH  = [XLabelHC{:}];
    set(XLabelH, 'String', [method_name ' iteration']) 
    % Save Figure
    fig_name = ['temporal_individual_tpt_' method_name];
    savefig(['./Output/' fig_name '.fig'])
    saveas(gcf,['./Output/' fig_name],'epsc')
    
    %% Average tpt experienced per WLAN
    mean_tpt_per_wlan = mean(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations,:),1);
    std_per_wlan = std(tptEvolutionPerWlan(minimumIterationToConsider:totalIterations,:),1);
    figure('pos',[450 400 500 350])
    axes;
    axis([1 20 30 70]);
    bar(mean_tpt_per_wlan, 0.5)
    set(gca, 'FontSize', 22)
    xlabel('WN id','fontsize', 24)
    ylabel('Mean throughput (Mbps)','fontsize', 24)
    hold on
    errorbar(mean_tpt_per_wlan, std_per_wlan, '.r');
    % Save Figure
    fig_name = ['mean_tpt_' method_name];
    savefig(['./Output/' fig_name '.fig'])
    saveas(gcf,['./Output/' fig_name],'epsc')
    
    %% Actions probability
    fig = figure('pos',[450 400 500 350]);
    axes;
    axis([1 20 30 70]);  
    % Print the preferred action per wlan
    for i=1:n_wlans             
        K = size(timesArmHasBeenPlayed, 2);
        subplot(2,2,i);
        bar(1:K, timesArmHasBeenPlayed(i, :)/totalIterations);
        hold on
        title(['WN' num2str(i)])
        axis([0 9 0 1])
        xticks(1:8)
        xticklabels(1:8)
        set(gca, 'FontSize', 22)
        % xticklabels({'ch=1/tpc=5','ch=2/tpc=5','ch=1/tpc=10','ch=2/tpc=10','ch=1/tpc=15','ch=2/tpc=15','ch=1/tpc=20','ch=2/tpc=20'})
    end
    % Set Axes labels
    AxesH    = findobj(fig, 'Type', 'Axes');       
    % Y-label
    YLabelHC = get(AxesH, 'YLabel');
    YLabelH  = [YLabelHC{:}];
    set(YLabelH, 'String', 'Action prob.', 'fontsize', 24)
    % X-label
    XLabelHC = get(AxesH, 'XLabel');
    XLabelH  = [XLabelHC{:}];
    set(XLabelH, 'String', 'Action index', 'fontsize', 24) 
    % Save Figure
    fig_name = ['actions_probability_' method_name];
    savefig(['./Output/' fig_name '.fig'])
    saveas(gcf,['./Output/' fig_name],'epsc')
    
end