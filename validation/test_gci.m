function test_gci()
%TEST_GCI  Unit test for gci_triplet.m
%
%  Feeds the GCI algorithm synthetic data with a KNOWN order of accuracy and
%  checks it recovers it. Construct f(h) = f_exact + C*h^p_true, hand the
%  triplet to gci_triplet, and require:
%     * the recovered p matches p_true
%     * the Richardson extrapolation recovers f_exact
%     * GCI is a sane positive band
%     * oscillatory and degenerate triplets are FLAGGED, not turned into
%       confident-looking numbers
%
%  Uses the same non-constant refinement ratios as the real Poiseuille mesh
%  series (r ~ 1.31 / 1.46), which is exactly the case the general iterative-p
%  solution exists for -- a constant-r shortcut would be wrong here.
%
%  This test is deliberately SELF-CONTAINED: it does not call
%  poiseuille_config and needs no exported data. It tests the algorithm, not
%  the dataset, so it must pass on a bare checkout.
%
%  Run:  test_gci
clc;
fprintf('\n=== gci_triplet unit test ===\n\n');

% h values of the real 36k / 62k / 135k meshes (h = sqrt(A/N), A = 1e-3 m^2,
% N = 36498 / 62746 / 134542). Hardcoded so the test has no data dependency;
% only their RATIOS matter (r21 = 1.464, r32 = 1.311).
h3 = 1.655257e-04;   % coarsest of the triplet (36k)
h2 = 1.262429e-04;   % medium               (62k)
h1 = 8.621266e-05;   % finest               (135k)

f_exact = -12000;
C = 5.0e6;
n_pass = 0; n_fail = 0;

for p_true = [1.0, 1.5, 2.0, 3.0]
    fprintf('--- synthetic data, p_true = %.2f ---\n', p_true);
    f3 = f_exact + C*h3^p_true;
    f2 = f_exact + C*h2^p_true;
    f1 = f_exact + C*h1^p_true;

    R = gci_triplet(f1, f2, f3, h1, h2, h3);
    fprintf('       status = %s,  r21 = %.4f, r32 = %.4f\n', R.status, R.r21, R.r32);
    check(sprintf('p recovered (p_true=%.2f)', p_true), R.p, p_true, 1e-6);
    check('Richardson f_ext -> f_exact', R.f_ext, f_exact, 1e-6);
    if R.gci_fine > 0 && R.gci_fine < 1
        pass(sprintf('GCI sane band (%.4f%%)', R.gci_fine*100));
    else
        fail('GCI sane band', R.gci_fine);
    end
    fprintf('\n');
end

fprintf('--- oscillatory (non-monotone) triplet ---\n');
R = gci_triplet(-12000+50, -12000-40, -12000+60, h1, h2, h3);
if contains(R.status, 'OSCILLATORY')
    pass(['oscillatory flagged: ' R.status]);
else
    fail(['oscillatory flagged, got: ' R.status], NaN);
end

fprintf('\n--- degenerate triplet (f1 == f2) ---\n');
R = gci_triplet(-12000, -12000, -12000, h1, h2, h3);
if contains(R.status, 'DEGENERATE') && isnan(R.p)
    pass(['degenerate flagged: ' R.status]);
else
    fail(['degenerate flagged, got: ' R.status], R.p);
end

fprintf('\n=====================================\n');
fprintf('  %d passed, %d failed\n', n_pass, n_fail);
fprintf('=====================================\n\n');

if n_fail > 0
    error('test_gci:failures', '%d GCI unit test(s) failed.', n_fail);
end

    function check(name, got, want, tol)
        if abs(got - want) <= tol*max(1, abs(want))
            pass(sprintf('%s  got %.8g', name, got));
        else
            fail(sprintf('%s (want %.8g)', name, want), got);
        end
    end

    function pass(msg)
        fprintf('  [PASS] %s\n', msg);
        n_pass = n_pass + 1;
    end

    function fail(msg, got)
        fprintf('  [FAIL] %s  got %g\n', msg, got);
        n_fail = n_fail + 1;
    end
end
