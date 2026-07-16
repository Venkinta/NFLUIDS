%% =====================================================================
%   RUN THE WHOLE POISEUILLE VALIDATION SUITE
%
%   This is the entry point. Type:   validate_all
%
%   Prerequisite: run validation/paraview_export_velocity_poiseuille.py in
%   ParaView first -- it produces the CSVs and poiseuille_meta.csv that every
%   script below reads. Without them this stops immediately with a message
%   naming the missing file (by design: no guessing, no stale data).
%
%   Runs, in order:
%     1. validate_poiseuille  - velocity: absolute/shape L2, mass error,
%                               u_max, symmetry, wall shear
%     2. validate_pressure    - dp/dx vs the fixed -12000 Pa/m, linearity
%     3. validate_gci         - grid convergence: observed order p, GCI
%     4. validate_diagnostics - mass vs x, entrance length, residual fields
%
%   Each is still runnable on its own by name.
%   To unit-test the GCI maths without any data:  test_gci
%  =====================================================================
clear; clc; close all;

scripts = {'validate_poiseuille', 'validate_pressure', ...
           'validate_gci', 'validate_diagnostics'};

fprintf('\n');
fprintf('#########################################################################\n');
fprintf('#   POISEUILLE VALIDATION SUITE                                          #\n');
fprintf('#########################################################################\n');

cfg = poiseuille_config();   % fails loudly here if data_dir/meta is wrong
fprintf('  data_dir : %s\n', cfg.data_dir);
fprintf('  meshes   : %s\n', strjoin(cfg.labels, ', '));
fprintf('  cells    : %s\n', mat2str(cfg.cells));
fprintf('  analytical anchors (mesh-independent, from the PRESCRIBED inlet):\n');
fprintf('     u_avg = %.4g m/s | dp/dx = %.1f Pa/m | u_max = %.4g m/s | tau_w = %.4g Pa\n', ...
    cfg.u_avg, cfg.dpdx_theory, cfg.u_max_theory, cfg.tau_w_theory);

for i = 1:numel(scripts)
    fprintf('\n');
    fprintf('#########################################################################\n');
    fprintf('#  [%d/%d]  %s\n', i, numel(scripts), scripts{i});
    fprintf('#########################################################################\n');
    run_isolated(scripts{i});
end

fprintf('\n');
fprintf('#########################################################################\n');
fprintf('#   SUITE COMPLETE -- %d scripts, %d figure windows\n', ...
    numel(scripts), numel(findobj('Type','figure')));
fprintf('#########################################################################\n\n');


function run_isolated(name)
%RUN_ISOLATED  Run a script inside this function's own workspace.
%   run() executes a script in the CALLER's workspace, so calling the
%   validators directly from the loop above would let their variables collide
%   with it -- both this file and validate_poiseuille use `i` as a loop index,
%   which would silently break the loop. A function body gives each script a
%   clean scope.
run(name);
end
