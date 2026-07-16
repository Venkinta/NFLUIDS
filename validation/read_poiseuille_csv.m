function d = read_poiseuille_csv(filepath)
%READ_POISEUILLE_CSV  Robust reader for a ParaView PlotOverLine CSV export.
%
%   d = READ_POISEUILLE_CSV(filepath) returns a struct with fields:
%       x, y      - sample coordinates (Points:0, Points:1)
%       u, v      - velocity components (Velocity:0, Velocity:1)
%       p         - Pressure
%       res_cont  - ContinuityResidual   (NaN-filled if absent)
%       res_mom   - MomentumResidual     (NaN-filled if absent)
%       arc       - arc_length
%       n_dropped - how many rows the valid-point mask removed
%
%   Rows with vtkValidPointMask == 0 are removed. This is not optional: a
%   sample point that lands exactly on a domain boundary falls outside every
%   cell and ParaView returns NaN for it (the channel inlet at x = 0.2365 does
%   exactly this). Leaving those rows in poisons every downstream mean, trapz
%   and polyfit with NaN.
%
%   Column names are matched defensively: ParaView writes "Velocity:0" but
%   MATLAB's readtable sanitises that to "Velocity_0", and the exact mangling
%   has changed between releases.

if ~exist(filepath, 'file')
    error('read_poiseuille_csv:notFound', ...
        ['File not found: %s\n' ...
         'Check cfg.data_dir in poiseuille_config.m, and make sure the ' ...
         'ParaView export has been run against the CURRENT solved VTUs.'], ...
        filepath);
end

% ParaView writes "Velocity:0"; readtable sanitises it to "Velocity_0" and
% warns every single time. The sanitised names are exactly what the matching
% below expects, so the warning is pure noise -- it would otherwise fire once
% per file and bury the actual report.
ws = warning('off', 'MATLAB:table:ModifiedAndSavedVarnames');
cleanup = onCleanup(@() warning(ws));
t = readtable(filepath);
vn = t.Properties.VariableNames;

    function col = pick(prefix, suffix, required)
        % Match e.g. prefix='Velocity', suffix='0' against Velocity:0 /
        % Velocity_0 / Velocity0, without also matching Points_0.
        hit = find(startsWith(vn, prefix) & ...
                   (endsWith(vn, ['_' suffix]) | endsWith(vn, [':' suffix]) | ...
                    strcmp(vn, [prefix suffix])), 1);
        if isempty(hit)
            if required
                error('read_poiseuille_csv:missingColumn', ...
                    'Could not find column %s:%s in %s\nAvailable: %s', ...
                    prefix, suffix, filepath, strjoin(vn, ', '));
            end
            col = [];
        else
            col = t.(vn{hit});
        end
    end

    function col = pick_exact(name, required)
        hit = find(strcmp(vn, name), 1);
        if isempty(hit)
            if required
                error('read_poiseuille_csv:missingColumn', ...
                    'Could not find column %s in %s\nAvailable: %s', ...
                    name, filepath, strjoin(vn, ', '));
            end
            col = [];
        else
            col = t.(vn{hit});
        end
    end

x = pick('Points', '0', true);
y = pick('Points', '1', true);
u = pick('Velocity', '0', true);
v = pick('Velocity', '1', false);
p = pick_exact('Pressure', true);
rc = pick_exact('ContinuityResidual', false);
rm = pick_exact('MomentumResidual', false);
arc = pick_exact('arc_length', false);

n_all = height(t);
if ismember('vtkValidPointMask', vn)
    valid = t.vtkValidPointMask == 1;
else
    warning('read_poiseuille_csv:noMask', ...
        'No vtkValidPointMask in %s; falling back to finite-value filter.', ...
        filepath);
    valid = isfinite(u) & isfinite(p);
end

% Even where the mask says "valid", a NaN can slip through; drop those too.
valid = valid & isfinite(x) & isfinite(y) & isfinite(u) & isfinite(p);

d.n_dropped = n_all - sum(valid);
d.x = x(valid);
d.y = y(valid);
d.u = u(valid);
d.p = p(valid);
if isempty(v),  d.v = nan(size(d.x));  else, d.v = v(valid);  end
if isempty(rc), d.res_cont = nan(size(d.x)); else, d.res_cont = rc(valid); end
if isempty(rm), d.res_mom = nan(size(d.x));  else, d.res_mom = rm(valid);  end
if isempty(arc), d.arc = nan(size(d.x)); else, d.arc = arc(valid); end

if isempty(d.x)
    error('read_poiseuille_csv:allInvalid', ...
        'Every row of %s was invalid/NaN.', filepath);
end
end
