%% =====================================================================
%                VELOCITY VALIDATION SUITE (POISEUILLE FLOW)
%
%  Reports THREE distinct error measures, because they answer three
%  different questions and a single "L2 error" conflates them:
%
%    ABSOLUTE L2  vs the true parabola (u_avg = 0.1 prescribed)
%                 -> "is my answer right?"                     <- the real test
%    SHAPE L2     vs a parabola rescaled to the CFD's own u_avg
%                 -> "is my profile parabolic?"    (shape only, flow-rate blind)
%    MASS ERROR   measured u_avg vs the prescribed 0.1
%                 -> "do I conserve mass?"
%
%  The previous version of this script reported only SHAPE L2 but labelled it
%  as validation against theory. Because it rebuilt the analytical curve from
%  the CFD's own integrated flow rate, a solver that manufactured 13% extra
%  mass could still score a good "error" -- the goalposts moved with it.
%  =====================================================================
% NOTE: deliberately no 'clear; clc; close all' -- validate_all runs the four
% validators in sequence, and clearing the console or closing figures would
% destroy the previous script's report and plots.

cfg = poiseuille_config();
n = numel(cfg.tags);

% Pre-allocate metrics
l2_abs    = zeros(n,1);   % absolute L2 vs true parabola          [%]
l2_shape  = zeros(n,1);   % shape-only L2 vs rescaled parabola    [%]
linf_abs  = zeros(n,1);   % max |u - u_theory|                    [m/s]
u_avg_m   = zeros(n,1);   % measured mean velocity                [m/s]
mass_err  = zeros(n,1);   % (u_avg_m/u_avg - 1)*100               [%]
u_max_m   = zeros(n,1);   % measured peak velocity                [m/s]
u_max_err = zeros(n,1);   % vs 1.5*u_avg                          [%]
asym      = zeros(n,1);   % top/bottom asymmetry                  [%]
tau_bot   = zeros(n,1);   % wall shear, bottom                    [Pa]
tau_top   = zeros(n,1);   % wall shear, top                       [Pa]
tau_err   = zeros(n,1);   % mean |tau| vs theory                  [%]

D = cell(n,1);   % cache each dataset -- the old script read every CSV twice

fig = figure('Color','w','Units','inches','Position',[1 1 13 8]);
sgtitle(sprintf(['Velocity Profile Validation at x = %.0f%% L  ' ...
    '(CFD vs Analytical, u_{avg} = %.3g m/s prescribed)'], 90, cfg.u_avg), ...
    'FontSize', 15, 'FontWeight','bold', 'FontName','Times New Roman');

for i = 1:n
    f = fullfile(cfg.data_dir, sprintf('poiseuille_profile_%s.csv', cfg.tags{i}));
    d = read_poiseuille_csv(f);

    % Sort bottom -> top
    [y, si] = sort(d.y);
    u = d.u(si);
    D{i} = struct('y', y, 'u', u, 'res_cont', d.res_cont(si), ...
                  'res_mom', d.res_mom(si));

    % --- Measured flow rate (mass-conservation diagnostic) ---
    % trapz integrates only over the sampled span. The mask can clip the
    % endpoints, but u -> 0 at the walls so the missing slivers contribute
    % ~1e-6 of the integral; dividing by the full H is correct.
    u_avg_m(i)  = trapz(y, u) / cfg.H;
    mass_err(i) = (u_avg_m(i)/cfg.u_avg - 1) * 100;

    % --- THE analytical profile: anchored to the PRESCRIBED u_avg ---
    eta = (y - cfg.y_start) / cfg.H;
    u_theory = 6*cfg.u_avg * eta .* (1 - eta);

    % --- Shape-only reference: same parabola rescaled to the CFD's flow rate
    u_shape = 6*u_avg_m(i) * eta .* (1 - eta);

    l2_abs(i)   = norm(u - u_theory) / norm(u_theory) * 100;
    l2_shape(i) = norm(u - u_shape)  / norm(u_shape)  * 100;
    linf_abs(i) = max(abs(u - u_theory));

    % --- Peak velocity ---
    u_max_m(i)   = max(u);
    u_max_err(i) = (u_max_m(i)/cfg.u_max_theory - 1) * 100;

    % --- Symmetry: compare the profile against its own mirror ---
    % Interpolate onto a symmetric grid, then mirror about the centreline.
    yq = linspace(min(y), max(y), 401)';
    uq = interp1(y, u, yq, 'linear');
    u_mirror = flipud(uq);
    asym(i) = norm(uq - u_mirror) / norm(uq) * 100;

    % --- Wall shear stress: tau_w = mu * |du/dy| at each wall ---
    % A QUADRATIC fit in wall-local coordinates s (distance from the wall),
    % evaluated at s = 0. Three deliberate choices:
    %   * quadratic, not linear: u is parabolic, so a straight-line fit over a
    %     near-wall band returns the band's MEAN slope, not the wall slope --
    %     for a 10% band that is 54 vs 60 s^-1, a systematic 10% underestimate.
    %     For an exact parabola the quadratic recovers du/dy|_wall exactly.
    %   * wall-local s, not raw y: y ~ 0.27 over a 0.001 m band makes the
    %     Vandermonde matrix hopelessly ill-conditioned. With s in [0, band]
    %     the fit is well posed and the linear coefficient IS du/dy at the wall.
    %   * a fit, not a 2-point difference: ParaView extrapolates cell-centre
    %     data to the wall and returns u ~ 3e-4 there instead of 0, so
    %     differencing the first two samples inherits that error directly.
    band = 0.10 * cfg.H;
    mb = y <= (cfg.y_start + band);
    mt = y >= (cfg.y_end   - band);
    if nnz(mb) >= 5
        sb = y(mb) - cfg.y_start;              % distance from bottom wall
        cb = polyfit(sb, u(mb), 2);            % [a b c] for a*s^2 + b*s + c
        tau_bot(i) = cfg.mu * abs(cb(2));      % du/ds at s = 0  ==  b
    else
        tau_bot(i) = NaN;
    end
    if nnz(mt) >= 5
        st = cfg.y_end - y(mt);                % distance from top wall
        ct = polyfit(st, u(mt), 2);
        tau_top(i) = cfg.mu * abs(ct(2));
    else
        tau_top(i) = NaN;
    end
    tau_err(i) = (mean([tau_bot(i) tau_top(i)])/cfg.tau_w_theory - 1) * 100;

    % --- Subplot ---
    subplot(2,3,i); hold on;
    plot(u, y, 'LineStyle','none', 'Marker', cfg.markers{i}, 'MarkerSize', 4, ...
        'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
        'DisplayName','CFD');
    ys = linspace(cfg.y_start, cfg.y_end, 300)';
    es = (ys - cfg.y_start)/cfg.H;
    plot(6*cfg.u_avg*es.*(1-es), ys, 'k-', 'LineWidth', 1.4, ...
        'DisplayName','Analytical');
    title(sprintf('%s  (abs L_2 = %.3f%%)', cfg.labels{i}, l2_abs(i)), ...
        'FontSize', 11, 'FontWeight','bold');
    xlabel('u  (m/s)','FontSize',9); ylabel('y  (m)','FontSize',9);
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    ylim([cfg.y_start cfg.y_end]);
    legend('Location','best','FontSize',8); hold off;
end

% --- Combined plot ---------------------------------------------------------
% Uses the cached data, and draws the ONE true analytical curve. The old
% version plotted whatever u_exact_smooth happened to survive the last loop
% iteration and labelled it "Theory".
subplot(2,3,6); hold on;
for i = 1:n
    y = D{i}.y; u = D{i}.u;
    ds = max(1, round(numel(y)/40));
    plot(u(1:ds:end), y(1:ds:end), 'LineStyle','none', ...
        'Marker', cfg.markers{i}, 'MarkerSize', 4, ...
        'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
        'DisplayName', cfg.labels{i});
end
ys = linspace(cfg.y_start, cfg.y_end, 300)';
es = (ys - cfg.y_start)/cfg.H;
plot(6*cfg.u_avg*es.*(1-es), ys, 'k-', 'LineWidth', 2, 'DisplayName','Analytical');
title('All Meshes Combined','FontSize',12,'FontWeight','bold');
xlabel('u  (m/s)','FontSize',9); ylabel('y  (m)','FontSize',9);
grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
ylim([cfg.y_start cfg.y_end]);
legend('Location','best','FontSize',8); hold off;

%% --- REPORT ---------------------------------------------------------------
fprintf('\n');
fprintf('=========================================================================================\n');
fprintf('                          VELOCITY VALIDATION REPORT                                     \n');
fprintf('   Analytical anchor: u_avg = %.4g m/s (PRESCRIBED at inlet, mesh-independent)\n', cfg.u_avg);
fprintf('   u_max_theory = %.4g m/s | tau_w_theory = %.4g Pa | Re_Dh = %.3g\n', ...
    cfg.u_max_theory, cfg.tau_w_theory, cfg.Re_Dh);
fprintf('=========================================================================================\n');
fprintf('  Mesh        |   Cells  | abs L2  | shape L2 | Max |err| | Mass err | u_max err\n');
fprintf('-----------------------------------------------------------------------------------------\n');
for i = 1:n
    fprintf('  %-11s | %7d  | %6.3f%% | %7.3f%% | %.3e | %+7.3f%% | %+7.3f%%\n', ...
        cfg.labels{i}, cfg.cells(i), l2_abs(i), l2_shape(i), linf_abs(i), ...
        mass_err(i), u_max_err(i));
end
fprintf('=========================================================================================\n');
fprintf('  abs L2   : vs true parabola (u_avg=%.3g). THE accuracy number.\n', cfg.u_avg);
fprintf('  shape L2 : vs parabola rescaled to each mesh''s own u_avg. Shape only --\n');
fprintf('             blind to flow-rate error. This is what the old script reported.\n');
fprintf('  Mass err : measured u_avg vs prescribed. NON-ZERO = CONTINUITY VIOLATION.\n');
fprintf('\n');
fprintf('=========================================================================================\n');
fprintf('                    WALL SHEAR + SYMMETRY (near-wall / BL quality)                       \n');
fprintf('=========================================================================================\n');
fprintf('  Mesh        | tau_bot  | tau_top  | mean err | asymmetry\n');
fprintf('-----------------------------------------------------------------------------------------\n');
for i = 1:n
    fprintf('  %-11s | %7.3f  | %7.3f  | %+7.3f%% | %7.4f%%\n', ...
        cfg.labels{i}, tau_bot(i), tau_top(i), tau_err(i), asym(i));
end
fprintf('=========================================================================================\n');
fprintf('  tau_w    : mu*|du/dy| at the wall, from a quadratic fit over the near-wall %.0f%% of H.\n', 100*0.10);
fprintf('             Directly probes boundary-layer resolution. Theory = %.4g Pa.\n', cfg.tau_w_theory);
fprintf('  asymmetry: profile vs its own mirror about the centreline. Should be ~0;\n');
fprintf('             non-zero implies a mesh or boundary-condition bias.\n');
fprintf('=========================================================================================\n\n');
