function R = gci_triplet(f1, f2, f3, h1, h2, h3)
%GCI_TRIPLET  Roache Grid Convergence Index for one mesh triplet.
%
%   R = GCI_TRIPLET(f1,f2,f3, h1,h2,h3) implements the procedure of
%   Celik et al. (2008), "Procedure for Estimation and Reporting of
%   Uncertainty Due to Discretization in CFD Applications", J. Fluids Eng.
%   130(7), 078001 -- the method behind ASME V&V 20.
%
%   INPUTS   f1 = solution on the FINEST grid (spacing h1)
%            f2 = medium (h2),  f3 = COARSEST (h3).   Requires h1 < h2 < h3.
%
%   OUTPUT struct R with fields:
%     r21, r32   refinement ratios h2/h1, h3/h2
%     p          observed order of accuracy
%     f_ext      Richardson-extrapolated (h -> 0) value
%     gci_fine   GCI on the finest grid, as a FRACTION (0.01 == 1%), Fs = 1.25
%     status     'monotone' | 'OSCILLATORY' | 'DEGENERATE ...'
%
%   The refinement ratios here are non-constant, so p is obtained from the
%   general fixed-point iteration
%       p = |ln|e32/e21| + q(p)| / ln(r21),
%       q(p) = ln( (r21^p - s) / (r32^p - s) ),   s = sign(e32/e21)
%   rather than the constant-r shortcut (for which q vanishes).
%
%   GCI is only meaningful for MONOTONE convergence inside the asymptotic
%   range. Oscillatory triplets (e32/e21 < 0) are flagged, not silently
%   turned into a confident-looking number.

R.r21 = h2/h1;
R.r32 = h3/h2;
R.p = NaN; R.f_ext = NaN; R.gci_fine = NaN;

e21 = f2 - f1;
e32 = f3 - f2;

scale = max(abs([f1 f2 f3]));
if scale == 0 || abs(e21) < 1e-14*scale
    % The two finest grids agree to machine precision: no discretisation
    % error left to measure (or an exact solution). p is undefined.
    R.status = 'DEGENERATE (f2==f1)';
    return
end

ratio = e32/e21;
s = sign(ratio);
if s > 0
    R.status = 'monotone';
else
    R.status = 'OSCILLATORY';
end

p = 2.0;
converged = false;
for it = 1:500
    numer = R.r21^p - s;
    denom = R.r32^p - s;
    if numer <= 0 || denom <= 0 || ~isfinite(numer) || ~isfinite(denom)
        break
    end
    q = log(numer/denom);
    p_new = abs(log(abs(ratio)) + q) / log(R.r21);
    if ~isfinite(p_new) || p_new > 20
        break
    end
    if abs(p_new - p) < 1e-10
        p = p_new; converged = true; break
    end
    p = 0.5*p + 0.5*p_new;   % damped, keeps the iteration stable
end

if ~converged
    R.status = [R.status ' / p-iter no-converge'];
    return
end
R.p = p;

rp = R.r21^p;
if abs(rp - 1) < 1e-12
    return
end
R.f_ext    = (rp*f1 - f2)/(rp - 1);
e_a        = abs((f1 - f2)/f1);
R.gci_fine = 1.25 * e_a / (rp - 1);
end
