%% =====================================================================
%              POISEUILLE DIAGNOSTICS  (where does the error live?)
%
%  Four things the profile/pressure scripts can't see:
%
%   1. MASS CONSERVATION vs x   - u_avg(x) must equal the prescribed 0.1 m/s at
%                                 EVERY station. Any drift is a continuity
%                                 violation. This is the check whose absence let
%                                 a +13.6% mass error hide inside "theory".
%   2. ENTRANCE LENGTH          - u_centreline(x) approaching 1.5*u_avg. Confirms
%                                 the profile station is genuinely developed and
%                                 lets us compare against Chen's correlation.
%   3. RESIDUAL FIELDS          - ContinuityResidual / MomentumResidual are
%                                 already exported in every CSV and were being
%                                 thrown away. Plotting them vs y and vs x says
%                                 WHERE the solver is struggling (near-wall? at
%                                 the corners? the outlet?).
%   4. PROFILE DEVELOPMENT      - the profile at each station, so you can watch
%                                 plug -> parabola.
%
%  Sections 1, 2 and 4 need the station/centreline CSVs from the updated
%  ParaView export. If those are absent the script says so and still runs 3.
%  =====================================================================
% NOTE: deliberately no 'clear; clc; close all' -- validate_all runs the four
% validators in sequence, and clearing the console or closing figures would
% destroy the previous script's report and plots.

cfg = poiseuille_config();
n = numel(cfg.tags);

have_stations = ~isempty(dir(fullfile(cfg.data_dir, 'poiseuille_station_*_*.csv')));
have_centre   = ~isempty(dir(fullfile(cfg.data_dir, 'poiseuille_centerline_*.csv')));

if ~have_stations || ~have_centre
    fprintf(['\n[!] Station/centreline CSVs not found in\n    %s\n' ...
        '    Sections 1, 2 and 4 need them -- re-run\n' ...
        '    paraview_export_velocity_poiseuille.py (the updated version)\n' ...
        '    against your CURRENT solved VTUs.\n' ...
        '    Running the residual-field section only.\n\n'], cfg.data_dir);
end

%% --- 1 + 4. MASS CONSERVATION AND PROFILE DEVELOPMENT vs x -------------
if have_stations
    fig1 = figure('Color','w','Units','inches','Position',[1 1 12 5]);
    sgtitle('Mass conservation along the channel', ...
        'FontSize', 14, 'FontWeight','bold','FontName','Times New Roman');

    fprintf('=========================================================================================\n');
    fprintf('                     MASS CONSERVATION vs x   (u_avg should be %.4g m/s)\n', cfg.u_avg);
    fprintf('=========================================================================================\n');

    subplot(1,2,1); hold on;
    all_worst = zeros(n,1);
    for i = 1:n
        files = dir(fullfile(cfg.data_dir, ...
            sprintf('poiseuille_station_%s_*.csv', cfg.tags{i})));
        if isempty(files), continue; end
        [~, ord] = sort({files.name});
        files = files(ord);

        xs = zeros(numel(files),1);
        ua = zeros(numel(files),1);
        for k = 1:numel(files)
            d = read_poiseuille_csv(fullfile(files(k).folder, files(k).name));
            [y, sj] = sort(d.y); u = d.u(sj);
            xs(k) = mean(d.x);                 % vertical line -> x is constant
            ua(k) = trapz(y, u)/cfg.H;
        end
        err = (ua/cfg.u_avg - 1)*100;
        all_worst(i) = max(abs(err));

        plot((xs - cfg.x_start)/cfg.L*100, err, '-', 'Color', cfg.colors{i}, ...
            'LineWidth', 1.2, 'Marker', cfg.markers{i}, 'MarkerSize', 5, ...
            'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
            'DisplayName', cfg.labels{i});

        fprintf('  %-11s  worst |mass error| = %+7.4f%%   (over %d stations)\n', ...
            cfg.labels{i}, all_worst(i), numel(files));
    end
    yline(0, 'k-', 'LineWidth', 1, 'HandleVisibility','off');
    xlabel('x  (% of channel length)','FontSize',10);
    ylabel('mass error  (%)','FontSize',10);
    title('u_{avg}(x) vs prescribed','FontSize',11,'FontWeight','bold');
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    legend('Location','best','FontSize',8); hold off;
    fprintf('=========================================================================================\n');
    fprintf('  Non-zero = the solver is creating or destroying mass at that station.\n');
    fprintf('  This is a hard conservation check: it does not depend on the profile shape.\n');
    fprintf('=========================================================================================\n\n');

    % --- Profile development, finest mesh ---
    subplot(1,2,2); hold on;
    i = n;   % finest
    files = dir(fullfile(cfg.data_dir, ...
        sprintf('poiseuille_station_%s_*.csv', cfg.tags{i})));
    [~, ord] = sort({files.name}); files = files(ord);
    % Sequential shading across stations = another ordinal ramp.
    greys = linspace(0.75, 0.0, numel(files))';
    for k = 1:numel(files)
        d = read_poiseuille_csv(fullfile(files(k).folder, files(k).name));
        [y, sj] = sort(d.y); u = d.u(sj);
        frac = (mean(d.x) - cfg.x_start)/cfg.L*100;
        plot(u, y, '-', 'Color', [1 1 1]*greys(k), 'LineWidth', 1.1, ...
            'DisplayName', sprintf('x = %.0f%% L', frac));
    end
    ys = linspace(cfg.y_start, cfg.y_end, 300)';
    es = (ys - cfg.y_start)/cfg.H;
    plot(6*cfg.u_avg*es.*(1-es), ys, 'r--', 'LineWidth', 1.6, ...
        'DisplayName','Analytical');
    xlabel('u  (m/s)','FontSize',10); ylabel('y  (m)','FontSize',10);
    title(sprintf('Profile development (%s)', cfg.labels{n}), ...
        'FontSize',11,'FontWeight','bold');
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    ylim([cfg.y_start cfg.y_end]);
    legend('Location','eastoutside','FontSize',7); hold off;
end

%% --- 2. ENTRANCE LENGTH ------------------------------------------------
if have_centre
    fig2 = figure('Color','w','Units','inches','Position',[1 1 11 4.5]);

    subplot(1,2,1); hold on;
    fprintf('=========================================================================================\n');
    fprintf('                              ENTRANCE LENGTH\n');
    fprintf('   Chen (1973) correlation at Re_Dh = %.3g:  L_e ~ %.5f m  (%.2f%% of L)\n', ...
        cfg.Re_Dh, cfg.L_entrance, cfg.L_entrance/cfg.L*100);
    fprintf('=========================================================================================\n');
    for i = 1:n
        f = fullfile(cfg.data_dir, sprintf('poiseuille_centerline_%s.csv', cfg.tags{i}));
        if ~exist(f,'file'), continue; end
        d = read_poiseuille_csv(f);
        [x, si] = sort(d.x); u = d.u(si);
        plot((x - cfg.x_start)*1000, u, '-', 'Color', cfg.colors{i}, ...
            'LineWidth', 1.3, 'DisplayName', cfg.labels{i});

        % L_e := first x where u_centreline reaches 99% of its developed value.
        u_dev = cfg.u_max_theory;
        idx = find(u >= 0.99*u_dev, 1, 'first');
        if isempty(idx)
            fprintf('  %-11s  never reaches 99%% of u_max_theory (peak %.5f m/s)\n', ...
                cfg.labels{i}, max(u));
        else
            Le = x(idx) - cfg.x_start;
            fprintf('  %-11s  L_e(99%%) = %.5f m  (%.2f%% of L)   [Chen: %.5f m]\n', ...
                cfg.labels{i}, Le, Le/cfg.L*100, cfg.L_entrance);
        end
    end
    yline(cfg.u_max_theory, 'k--', 'LineWidth', 1.2, 'DisplayName','1.5 u_{avg}');
    xline(cfg.L_entrance*1000, 'r:', 'LineWidth', 1.4, 'DisplayName','L_e (Chen)');
    xlabel('distance from inlet  (mm)','FontSize',10);
    ylabel('u on centreline  (m/s)','FontSize',10);
    title('Entrance development','FontSize',11,'FontWeight','bold');
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    legend('Location','best','FontSize',8); hold off;
    fprintf('=========================================================================================\n');
    fprintf('  The profile station (90%% L = %.1f mm) must be well past L_e for the\n', 0.9*cfg.L*1000);
    fprintf('  fully-developed parabola to be the right reference. It is.\n');
    fprintf('=========================================================================================\n\n');

    % Zoom on the entrance
    subplot(1,2,2); hold on;
    for i = 1:n
        f = fullfile(cfg.data_dir, sprintf('poiseuille_centerline_%s.csv', cfg.tags{i}));
        if ~exist(f,'file'), continue; end
        d = read_poiseuille_csv(f);
        [x, si] = sort(d.x); u = d.u(si);
        plot((x - cfg.x_start)*1000, u, '-', 'Color', cfg.colors{i}, ...
            'LineWidth', 1.3, 'DisplayName', cfg.labels{i});
    end
    yline(cfg.u_max_theory, 'k--','LineWidth',1.2,'HandleVisibility','off');
    xline(cfg.L_entrance*1000, 'r:', 'LineWidth',1.4,'HandleVisibility','off');
    xlim([0 max(3*cfg.L_entrance*1000, 5)]);
    xlabel('distance from inlet  (mm)','FontSize',10);
    ylabel('u on centreline  (m/s)','FontSize',10);
    title('Entrance zoom','FontSize',11,'FontWeight','bold');
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    legend('Location','best','FontSize',8); hold off;
end

%% --- 3. RESIDUAL FIELDS ------------------------------------------------
% These columns ride along in every export and were previously discarded.
fig3 = figure('Color','w','Units','inches','Position',[1 1 12 8]);
sgtitle('Residual distribution: where is the solver working hardest?', ...
    'FontSize', 14, 'FontWeight','bold','FontName','Times New Roman');

fprintf('=========================================================================================\n');
fprintf('                            RESIDUAL FIELD SUMMARY\n');
fprintf('=========================================================================================\n');
fprintf('  Mesh        | cont (profile) max/mean | mom (profile) max/mean | cont (centre) max\n');
fprintf('-----------------------------------------------------------------------------------------\n');

for i = 1:n
    dv = read_poiseuille_csv(fullfile(cfg.data_dir, ...
        sprintf('poiseuille_profile_%s.csv', cfg.tags{i})));
    [y, sj] = sort(dv.y);
    rc = dv.res_cont(sj); rm = dv.res_mom(sj);

    dp = read_poiseuille_csv(fullfile(cfg.data_dir, ...
        sprintf('poiseuille_pressure_%s.csv', cfg.tags{i})));
    [x, si] = sort(dp.x);
    rcx = dp.res_cont(si);

    fprintf('  %-11s |   %.3e / %.3e   |  %.3e / %.3e  |  %.3e\n', ...
        cfg.labels{i}, max(rc), mean(rc), max(rm), mean(rm), max(rcx));

    % Continuity residual across the channel (y)
    subplot(2,2,1); hold on;
    semilogy(rc, y, '-', 'Color', cfg.colors{i}, 'LineWidth', 1.2, ...
        'DisplayName', cfg.labels{i});

    % Momentum residual across the channel (y)
    subplot(2,2,2); hold on;
    semilogy(rm, y, '-', 'Color', cfg.colors{i}, 'LineWidth', 1.2, ...
        'DisplayName', cfg.labels{i});

    % Continuity residual along the centreline (x)
    subplot(2,2,3); hold on;
    semilogy((x - cfg.x_start)/cfg.L*100, rcx, '-', 'Color', cfg.colors{i}, ...
        'LineWidth', 1.2, 'DisplayName', cfg.labels{i});

    subplot(2,2,4); hold on;
    semilogy((x - cfg.x_start)/cfg.L*100, dp.res_mom(si), '-', ...
        'Color', cfg.colors{i}, 'LineWidth', 1.2, 'DisplayName', cfg.labels{i});
end
fprintf('=========================================================================================\n');
fprintf('  A residual peak hugging the wall points at boundary-layer cells;\n');
fprintf('  a peak at x ~ 0%% or 100%% points at the inlet/outlet corners.\n');
fprintf('=========================================================================================\n\n');

subplot(2,2,1);
set(gca,'XScale','log');
xlabel('ContinuityResidual','FontSize',9); ylabel('y  (m)','FontSize',9);
title('Continuity residual across channel','FontSize',11,'FontWeight','bold');
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
ylim([cfg.y_start cfg.y_end]); legend('Location','best','FontSize',7); hold off;

subplot(2,2,2);
set(gca,'XScale','log');
xlabel('MomentumResidual','FontSize',9); ylabel('y  (m)','FontSize',9);
title('Momentum residual across channel','FontSize',11,'FontWeight','bold');
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
ylim([cfg.y_start cfg.y_end]); legend('Location','best','FontSize',7); hold off;

subplot(2,2,3);
set(gca,'YScale','log');
xlabel('x  (% of L)','FontSize',9); ylabel('ContinuityResidual','FontSize',9);
title('Continuity residual along centreline','FontSize',11,'FontWeight','bold');
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
legend('Location','best','FontSize',7); hold off;

subplot(2,2,4);
set(gca,'YScale','log');
xlabel('x  (% of L)','FontSize',9); ylabel('MomentumResidual','FontSize',9);
title('Momentum residual along centreline','FontSize',11,'FontWeight','bold');
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
legend('Location','best','FontSize',7); hold off;
