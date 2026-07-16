%% =====================================================================
%           GRID CONVERGENCE STUDY  (Roache GCI / ASME V&V 20)
%
%  Computes the OBSERVED order of accuracy p and the Grid Convergence Index
%  for each consecutive mesh triplet, following Celik et al. (2008),
%  "Procedure for Estimation and Reporting of Uncertainty Due to
%  Discretization in CFD Applications", J. Fluids Eng. 130(7).
%
%  This is the standard verification result -- it says more than any single
%  error percentage, because it answers: is the solver converging at the rate
%  the discretisation promises, and what is the numerical uncertainty on the
%  finest answer?
%
%  Representative cell size in 2-D:  h = sqrt(A/N),  A = L*H.
%  The refinement ratios here are NON-CONSTANT (r ~ 1.31 to 1.91), so the
%  general iterative solution for p is used, not the constant-r shortcut.
%
%  IMPORTANT: GCI is only meaningful in the asymptotic range with MONOTONE
%  convergence. This script explicitly flags oscillatory triplets rather than
%  reporting a confident-looking p computed from noise.
%  =====================================================================
% NOTE: deliberately no 'clear; clc; close all' -- validate_all runs the four
% validators in sequence, and clearing the console or closing figures would
% destroy the previous script's report and plots.

cfg = poiseuille_config();
n = numel(cfg.tags);

% --- 1. Gather the functionals from the CSVs ----------------------------
% Each is a scalar "solution functional" whose grid convergence we assess.
dpdx  = zeros(n,1);
u_max = zeros(n,1);
u_avg = zeros(n,1);
tau_w = zeros(n,1);

for i = 1:n
    % Pressure gradient, developed window
    dp = read_poiseuille_csv(fullfile(cfg.data_dir, ...
        sprintf('poiseuille_pressure_%s.csv', cfg.tags{i})));
    [x, si] = sort(dp.x); p = dp.p(si);
    win = (x >= cfg.fit_x_lo) & (x <= cfg.fit_x_hi);
    cf = polyfit(x(win), p(win), 1);
    dpdx(i) = cf(1);

    % Velocity functionals
    dv = read_poiseuille_csv(fullfile(cfg.data_dir, ...
        sprintf('poiseuille_profile_%s.csv', cfg.tags{i})));
    [y, sj] = sort(dv.y); u = dv.u(sj);
    u_max(i) = max(u);
    u_avg(i) = trapz(y, u)/cfg.H;
    % Quadratic fit in wall-local coordinates -- see validate_poiseuille.m for
    % why a linear fit here biases tau_w low by ~10%.
    band = 0.10*cfg.H;
    mb = y <= (cfg.y_start + band);
    mt = y >= (cfg.y_end   - band);
    cb = polyfit(y(mb) - cfg.y_start, u(mb), 2);
    ct = polyfit(cfg.y_end - y(mt),   u(mt), 2);
    tau_w(i) = cfg.mu * mean([abs(cb(2)) abs(ct(2))]);
end

h = cfg.h(:);   % h(1) is the COARSEST (3k), h(end) the FINEST

quantities = { ...
    'dp/dx  [Pa/m]', dpdx,  cfg.dpdx_theory; ...
    'u_max  [m/s]',  u_max, cfg.u_max_theory; ...
    'u_avg  [m/s]',  u_avg, cfg.u_avg; ...
    'tau_w  [Pa]',   tau_w, cfg.tau_w_theory};

%% --- 2. Report the raw series ------------------------------------------
fprintf('\n');
fprintf('=========================================================================================\n');
fprintf('                       GRID CONVERGENCE STUDY (Roache GCI)                               \n');
fprintf('=========================================================================================\n');
fprintf('  Mesh        |   Cells  |   h = sqrt(A/N)  |   dp/dx    |  u_max   |  u_avg   |  tau_w\n');
fprintf('-----------------------------------------------------------------------------------------\n');
for i = 1:n
    fprintf('  %-11s | %7d  |   %.6e  | %10.2f | %.6f | %.6f | %7.3f\n', ...
        cfg.labels{i}, cfg.cells(i), h(i), dpdx(i), u_max(i), u_avg(i), tau_w(i));
end
fprintf('  %-11s |    ---   |        0         | %10.2f | %.6f | %.6f | %7.3f   <- ANALYTICAL\n', ...
    'exact', cfg.dpdx_theory, cfg.u_max_theory, cfg.u_avg, cfg.tau_w_theory);
fprintf('=========================================================================================\n');

%% --- 3. GCI per triplet -------------------------------------------------
for q = 1:size(quantities,1)
    name = quantities{q,1};
    f    = quantities{q,2};
    fex  = quantities{q,3};

    fprintf('\n-----------------------------------------------------------------------------------------\n');
    fprintf('  %s   (analytical = %.6g)\n', name, fex);
    fprintf('-----------------------------------------------------------------------------------------\n');
    fprintf('  Triplet (coarse->fine) |   r21   r32  |    p    | f_extrap  | GCI_fine | status\n');

    % Triplets are formed ONLY within a mesh family. A triplet spanning the
    % tri and quad families would compare different TOPOLOGIES, not different
    % resolutions -- the quad/triangle seam contributes a systematic wall-drag
    % error that is not a function of h, so Richardson would "measure" the
    % topology change and report a confident, meaningless p.
    fams = unique(cfg.series, 'stable');
    for s = 1:numel(fams)
        idx = find(strcmp(cfg.series, fams{s}));
        [~, ord] = sort(h(idx), 'descend');   % coarsest -> finest
        idx = idx(ord);
        if numel(idx) < 3
            fprintf('  [%s family: only %d mesh(es) -- a GCI triplet needs 3, skipping]\n', ...
                fams{s}, numel(idx));
            continue
        end
        for t = 1:(numel(idx) - 2)
            i3 = idx(t); i2 = idx(t+1); i1 = idx(t+2);   % i1 = finest
            R = gci_triplet(f(i1), f(i2), f(i3), h(i1), h(i2), h(i3));
            fprintf('  %-5s %-5s %-8s | %5.3f %5.3f  | %7s | %9s | %8s | %s\n', ...
                cfg.tags{i3}, cfg.tags{i2}, cfg.tags{i1}, R.r21, R.r32, ...
                fmt(R.p, '%7.3f'), fmt(R.f_ext, '%9.4g'), ...
                fmtpct(R.gci_fine), R.status);
        end
    end
end

fprintf('\n=========================================================================================\n');
fprintf('  p        : observed order of accuracy. A 2nd-order scheme should approach p ~ 2.\n');
fprintf('             Upwind convection + a non-orthogonal correction typically lands 1 < p < 2.\n');
fprintf('  f_extrap : Richardson-extrapolated h->0 value. Compare against ANALYTICAL above --\n');
fprintf('             agreement means the remaining gap is discretisation error, not a bug.\n');
fprintf('  GCI_fine : numerical uncertainty band on the FINEST mesh of the triplet (Fs = 1.25).\n');
fprintf('             Quote this as the error bar on your converged result.\n');
fprintf('  status   : "monotone" = usable. "OSCILLATORY" = the three solutions do not bracket\n');
fprintf('             a trend; p is not meaningful there and the value is reported for\n');
fprintf('             completeness only. Oscillatory convergence usually means the meshes are\n');
fprintf('             not in the asymptotic range, or that mesh topology (not just h) changed.\n');
fprintf('=========================================================================================\n\n');

%% --- 4. Convergence plot -----------------------------------------------
fig = figure('Color','w','Units','inches','Position',[1 1 12 8]);
sgtitle('Grid Convergence: error vs representative cell size h', ...
    'FontSize', 15, 'FontWeight','bold','FontName','Times New Roman');

for q = 1:size(quantities,1)
    name = quantities{q,1};
    f    = quantities{q,2};
    fex  = quantities{q,3};
    err  = abs(f - fex) / abs(fex) * 100;

    subplot(2,2,q); hold on;
    % Per-point markers in the ordinal ramp so mesh identity stays readable.
    for i = 1:n
        loglog(h(i), max(err(i), eps), 'LineStyle','none', ...
            'Marker', cfg.markers{i}, 'MarkerSize', 8, ...
            'MarkerFaceColor', cfg.colors{i}, 'MarkerEdgeColor','none', ...
            'DisplayName', cfg.labels{i});
    end
    % Connect points WITHIN a family only -- a line joining the tri and quad
    % series would imply a convergence trend between two different topologies,
    % which is exactly the false reading this whole study had to unlearn.
    fams = unique(cfg.series, 'stable');
    for s = 1:numel(fams)
        idx = find(strcmp(cfg.series, fams{s}));
        [hs, ord] = sort(h(idx));
        loglog(hs, max(err(idx(ord)), eps), '-', 'Color', [0.6 0.6 0.6], ...
            'LineWidth', 1, 'HandleVisibility','off');
    end

    % Reference slopes anchored at the finest point.
    hr = logspace(log10(min(h)), log10(max(h)), 50);
    a1 = max(err(end),eps) / h(end);
    a2 = max(err(end),eps) / h(end)^2;
    loglog(hr, a1*hr,    'k--', 'LineWidth', 1, 'DisplayName','1st order');
    loglog(hr, a2*hr.^2, 'k:',  'LineWidth', 1.2, 'DisplayName','2nd order');

    set(gca,'XScale','log','YScale','log');
    title(name, 'FontSize', 11, 'FontWeight','bold');
    xlabel('h  (m)','FontSize',9); ylabel('|error| vs analytical  (%)','FontSize',9);
    grid on; set(gca,'GridLineStyle',':','GridAlpha',0.35,'FontName','Times New Roman');
    legend('Location','best','FontSize',7);
    hold off;
end

%% --- local functions ----------------------------------------------------
function s = fmt(v, f)
if isnan(v), s = '  --   '; else, s = sprintf(f, v); end
end

function s = fmtpct(v)
if isnan(v), s = '   --   '; else, s = sprintf('%7.3f%%', v*100); end
end

% gci_triplet() lives in its own file (gci_triplet.m) so it can be unit-tested
% independently -- see test_gci.m, which checks it recovers a known order of
% accuracy from synthetic power-law data.
