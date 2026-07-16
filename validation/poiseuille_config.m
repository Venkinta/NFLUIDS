function cfg = poiseuille_config()
%POISEUILLE_CONFIG  Single source of truth for the Poiseuille validation suite.
%
%   Every validate_*.m script pulls its constants from here. If you change the
%   solver's fluid properties or the CAD rectangle, change them ONCE, here.
%
%   THE CENTRAL RULE OF THIS FILE
%   ----------------------------
%   The analytical solution is anchored to the PRESCRIBED inlet velocity, never
%   to the CFD result. u_avg is imposed as a uniform 0.1 m/s at the inlet;
%   conservation of mass makes it 0.1 m/s at EVERY x station. So dp/dx, u_max
%   and tau_w below are CONSTANTS -- identical for all five meshes.
%
%   An earlier version of this suite computed u_avg by integrating the CFD
%   profile and built "theory" from that. This is circular: it grades the solver
%   against itself. Its symptom was a "Theory dp/dx" column that varied from
%   -11543 to -13626 across meshes -- theory cannot depend on which mesh you
%   ran. Worse, it silently absorbed mass-conservation errors (the 62k mesh was
%   carrying +13.6% too much mass; that showed up as "theory moved", not as a
%   bug). The measured u_avg is still computed, but it is reported as a
%   MASS-CONSERVATION DIAGNOSTIC, which is what it actually is.

% =========================================================================
%  >>> THE ONLY LINE YOU EVER NEED TO EDIT <<<
%  Where the exported CSVs and solved .vtu files live. No searching, no
%  fallbacks: if this is wrong, every script stops immediately and says so.
%  (Data is deliberately kept OUTSIDE the repo -- ~107 MB of regenerable
%  output that must never be committed.)
% =========================================================================
cfg.data_dir = 'C:\Users\nesti\Documents\PYTHON_LEARNING\Meshes\Poiseuille\';
% =========================================================================

if ~isfolder(cfg.data_dir)
    error('poiseuille_config:noDataDir', ...
        ['Data directory does not exist:\n    %s\n' ...
         'Edit cfg.data_dir at the top of poiseuille_config.m.'], cfg.data_dir);
end

% --- Fluid properties (must match PhysicsEditor / the solver run) --------
cfg.rho = 1000;      % kg/m^3
cfg.mu  = 1.0;       % Pa*s  dynamic viscosity.
                     % (This case is mu = 1, NOT water -- water is 1e-3 Pa*s.)

% --- Geometry, from the CAD rectangle (SI metres) ------------------------
cfg.x_start = 0.2365;   % inlet
cfg.x_end   = 0.3365;   % outlet
cfg.y_start = 0.27;     % bottom wall
cfg.y_end   = 0.28;     % top wall
cfg.L = cfg.x_end - cfg.x_start;   % 0.1  m  channel length
cfg.H = cfg.y_end - cfg.y_start;   % 0.01 m  channel height
cfg.y_mid = 0.5*(cfg.y_start + cfg.y_end);

% --- THE analytical anchor: the prescribed inlet condition ---------------
cfg.u_avg = 0.1;                                      % m/s, IMPOSED at inlet

cfg.dpdx_theory  = -12*cfg.mu*cfg.u_avg / cfg.H^2;    % -12000 Pa/m
cfg.u_max_theory =  1.5*cfg.u_avg;                    %  0.15  m/s
cfg.tau_w_theory =   6*cfg.mu*cfg.u_avg / cfg.H;      %  60    Pa
% Sanity: the pressure force balances shear on both walls,
%   -dpdx*H == 2*tau_w   ->   12000*0.01 == 2*60 == 120 N/m^2.  OK.

% --- Reynolds number and entrance length ---------------------------------
cfg.Dh    = 2*cfg.H;                                   % hydraulic dia, 0.02 m
cfg.Re_Dh = cfg.rho*cfg.u_avg*cfg.Dh / cfg.mu;         % = 2  (deeply laminar)
% Chen (1973) laminar entrance-length correlation. The plain 0.05*Re*Dh form
% collapses to ~2 mm at Re=2 and badly underestimates: at creeping-flow Re the
% viscous-diffusion term dominates. Chen keeps the constant term:
cfg.L_entrance = cfg.Dh * (0.63/(1 + 0.035*cfg.Re_Dh) + 0.044*cfg.Re_Dh);
% ~= 0.0135 m, i.e. ~13.5% of the channel is still developing.

% --- Pressure fit window: developed region ONLY --------------------------
% Fitting dp/dx over the full channel includes the developing entrance, where
% the profile is still plug-like and the wall shear (hence the slope) is
% steeper. At Re_Dh = 2 the excess is small (Hagenbach K~0.67 gives
% 0.67*0.5*rho*u^2 ~ 3.35 Pa against a ~1200 Pa total drop, i.e. ~0.3%), but
% trimming also lets us SEE the non-linearity instead of smearing it into the
% slope. 20% inlet margin is ~1.5x L_entrance; 10% outlet margin is 1H.
cfg.fit_x_lo = cfg.x_start + 0.20*cfg.L;   % 0.2565
cfg.fit_x_hi = cfg.x_end   - 0.10*cfg.L;   % 0.3265

% --- Mesh series ---------------------------------------------------------
% TWO INDEPENDENT FAMILIES, and the distinction is load-bearing:
%   'tri'  - prismatic boundary layer + unstructured interior triangles
%   'quad' - quad-dominant (~92-96% quads; one unavoidable central strip)
% They are DIFFERENT TOPOLOGIES, not different resolutions of one mesh. Never
% compare across them with Richardson/GCI: the 2026-07 experiment showed the
% quad/triangle seam injects a systematic ~+4.5% wall-drag error that is NOT a
% function of h, so a triplet spanning both families measures topology, not
% order of accuracy. cfg.series keeps validate_gci from doing that.
cfg.tags    = {'3k', '10k', '36k', '62k', '135k', 'quads', 'quads_2'};
cfg.labels  = {'3k Cells', '10k Cells', '36k Cells', '62k Cells', '135k Cells', ...
               'Quad ~3k', 'Quad ~6k'};
cfg.series  = {'tri', 'tri', 'tri', 'tri', 'tri', 'quad', 'quad'};
cfg.markers = {'o', 's', '^', 'd', 'p', 'v', '>'};

% Mesh level is ORDINAL within a family: 3k->135k is a refinement sequence and
% swapping the order would change the meaning. Ordinal data takes a single-hue
% ramp with monotone lightness so the reader sees "coarse -> fine" in the colour
% itself. FAMILY is nominal, so it gets a different HUE: blues for 'tri',
% oranges for 'quad'. Both ramps validated (monotone lightness, adjacent
% dL >= 0.06, light-end contrast >= 2.0:1, single hue). The distinct markers
% above carry identity as secondary encoding.
cfg.colors = {'#6BAED6', '#4292C6', '#2171B5', '#08519C', '#08306B', ...
              '#FD8D3C', '#A63603'};

% --- Real cell counts ----------------------------------------------------
cfg.cells = local_cell_counts(cfg);

% --- Derived: representative cell size for grid convergence ---------------
% 2-D: h = sqrt(A/N) with A the fluid area. Proportional to 1/sqrt(N).
cfg.area = cfg.L * cfg.H;
cfg.h    = sqrt(cfg.area ./ cfg.cells(:));

end


function n = local_cell_counts(cfg)
%LOCAL_CELL_COUNTS  True per-mesh cell counts, read from poiseuille_meta.csv.
%
%   NO FALLBACK, deliberately. An earlier version warned and substituted
%   nominal counts when the meta file was missing. That is not a convenience,
%   it is a trap: cfg.h = sqrt(area./cells) feeds the grid-convergence study,
%   so wrong counts silently corrupt the observed order of accuracy p and the
%   GCI -- producing confident, wrong, thesis-grade numbers. Missing data must
%   stop the run, not be guessed around.
meta_path = fullfile(cfg.data_dir, 'poiseuille_meta.csv');

if ~exist(meta_path, 'file')
    error('poiseuille_config:noMeta', ...
        ['poiseuille_meta.csv not found in\n    %s\n' ...
         'Run validation/paraview_export_velocity_poiseuille.py against the ' ...
         'current solved VTUs -- it writes the true cell counts that the GCI ' ...
         'depends on.'], cfg.data_dir);
end

t = readtable(meta_path);
n = zeros(1, numel(cfg.tags));
for i = 1:numel(cfg.tags)
    row = strcmp(string(t.tag), cfg.tags{i});
    if ~any(row)
        error('poiseuille_config:missingTag', ...
            ['Mesh tag "%s" is absent from\n    %s\n' ...
             'Re-run the ParaView export so every mesh in cfg.tags is ' ...
             'represented.'], cfg.tags{i}, meta_path);
    end
    n(i) = t.n_cells(find(row, 1));
end
end
