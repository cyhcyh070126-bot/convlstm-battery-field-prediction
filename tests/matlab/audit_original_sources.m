function audit_original_sources()
repo = fileparts(fileparts(fileparts(mfilename('fullpath'))));
audit_dir = fullfile(repo, 'outputs', 'source-audit-matlab');
if ~isfolder(audit_dir), mkdir(audit_dir); end
addpath(fullfile(repo, 'matlab'));
addpath(fullfile(repo, 'matlab', 'geometry'));
cleanup_path = onCleanup(@() rmpath(fullfile(repo, 'matlab'), fullfile(repo, 'matlab', 'geometry'))); %#ok<NASGU>
report = struct();
checks = struct('file', {}, 'scope', {}, 'messages', {});
for scope = {'source_archive', 'research_scripts', 'matlab'}
    files = dir(fullfile(repo, scope{1}, '**', '*.m'));
    for i = 1:numel(files)
        filename = fullfile(files(i).folder, files(i).name);
        checks(end+1) = struct('file', filename, 'scope', scope{1}, ...
            'messages', checkcode(filename, '-id')); %#ok<AGROW>
    end
end
report.checkcode = checks;
original = fileread(fullfile(repo, 'source_archive', 'main_D21origin.m'));
packaged = fileread(fullfile(repo, 'matlab', 'geometry', 'main_D21origin.m'));
marker = 'Rs = Rs(:); % 确保最终Rs是列向量';
original = original(1:strfind(original, marker)+length(marker)-1);
packaged = packaged(1:strfind(packaged, marker)+length(marker)-1);
scratch = fullfile(audit_dir, 'matlab-fixture');
scratch = strrep(scratch, '\', '/');
if ~isfolder(scratch), mkdir(scratch); end
% Reproduce the packaging error without running the costly packing loop.
baseline = packaged(1:strfind(packaged, '% ---------------------------- Generate Radius Distribution')-1);
try
    isolated_prefix(baseline, scratch);
    report.standalone_prefix = 'passed';
catch err
    report.standalone_prefix = err.message;
end
results = struct('N',{}, 'command',{}, 'maximum_radius_difference',{}, 'count',{}, 'positive',{});
for count = [40,60,80]
    for branch = 1:3
        reference = source_radii(original, scratch, count, branch, 1234);
        actual = packaged_radii(packaged, scratch, count, branch, 1234);
        assert(isequal(reference, actual), 'Original and packaged radii differ.');
        assert(numel(actual) == count && all(isfinite(actual) & actual > 0));
        results(end+1) = struct('N', count, 'command', branch, ...
            'maximum_radius_difference', max(abs(reference-actual)), ...
            'count', numel(actual), 'positive', true); %#ok<AGROW>
        fprintf('N=%d branch=%d: exact original radii parity passed\n', count, branch);
    end
end
report.radius_parity = results;
report.options = validate_geometry_options(struct('N', 80, 'command', 2, 'mu', 2, 'sigma', .15));
for invalid = {struct('N', 2), struct('command', 4), struct('sigma', 0), struct('mu', NaN), struct('made_up', 1)}
    rejected = false;
    try, validate_geometry_options(invalid{1}); catch, rejected = true; end
    assert(rejected, 'Invalid geometry setting was accepted.');
end
report.invalid_options_rejected = 5;
% Reproduce then verify the original undefined-ax error using real MATLAB.
old_dir = pwd; cd(scratch); cwd_cleanup = onCleanup(@() cd(old_dir)); %#ok<NASGU>
draw_dir = fullfile(repo, 'research_scripts', '新版');
addpath(draw_dir, '-begin');
fig_cleanup = onCleanup(@() close('all')); %#ok<NASGU>
try
    Draw_Polycrystal([0,0,1.5; 1,0,1.5; 0,1,1.5; 1,1,1.5; .5,.5,1.5], 2);
    report.new_draw_ax = 'passed';
catch err
    report.new_draw_ax = err.message;
end
rmpath(draw_dir);
addpath(fullfile(repo, 'research_scripts'), '-begin');
for dimensions = [2,3]
    coordinates = 2*ones(1, dimensions);
    radius = norm(coordinates);
    original_direction = coordinates/radius;
    inner = Boundary_Particle_back_circle_2([coordinates,.2], 1, dimensions);
    assert(abs(norm(inner(1:dimensions))-.8) < 1e-12);
    assert(max(abs(inner(1:dimensions)/.8-original_direction)) < 1e-12);
    origin = Boundary_Particle_back_circle_2([zeros(1,dimensions),.2], 1, dimensions);
    assert(all(isfinite(origin)));
    wrapped = Boundary_Particle_back_circle([coordinates,.2], 1, dimensions);
    assert(norm(wrapped(1:dimensions)) <= 1.2+1e-12);
    alignment = dot(wrapped(1:dimensions)/norm(wrapped(1:dimensions)),original_direction);
    assert(abs(abs(alignment)-1) < 1e-12);
end
report.circular_boundary_checks = '2D and 3D projection, direction and origin passed';
invalid_boundary_rejected = false;
try, Circle_Boundary(.1,2); catch, invalid_boundary_rejected = true; end
assert(invalid_boundary_rejected);
report.small_boundary_rejected = true;
rmpath(fullfile(repo, 'research_scripts'));
addpath(fullfile(repo, 'tests', 'matlab'));
export_results = runtests(fullfile(repo, 'tests', 'matlab', 'test_export_validation.m'));
report.export_tests_passed = all([export_results.Passed]);
report.export_test_count = numel(export_results);
assert(report.export_tests_passed);
report.livelink_function_available = exist('mphsave','file') ~= 0;
report.matlab_version = version;
out = fopen(fullfile(audit_dir, 'matlab-results.json'), 'w', 'n', 'UTF-8');
file_cleanup = onCleanup(@() fclose(out)); %#ok<NASGU>
fprintf(out, '%s\n', jsonencode(report, PrettyPrint=true));
end

function isolated_prefix(code, scratch)
code = strrep(code, "fullfile(pwd, 'temp_figs')", "fullfile('"+string(scratch)+"', 'temp_figs')");
eval(code);
end

function radii = source_radii(code, scratch, count, branch, seed)
% Only replace external setup and original parameter assignments in this
% test harness; the real sampling and plotting code is evaluated verbatim.
code = regexprep(code, '(?m)^clear\s*$', '');
code = regexprep(code, '(?m)^clc\s*$', '');
code = regexprep(code, '(?m)^close all\s*$', '');
code = strrep(code, "setenv('BLAS_VERSION','')", '');
code = regexprep(code, 'temp_fig_folder = fullfile\([^\n]+', ...
    "temp_fig_folder = fullfile('"+string(scratch)+"', 'temp_figs');");
code = strrep(code, 'N=60;', sprintf('N=%d;',count));
code = strrep(code, 'command=3;', sprintf('command=%d;',branch));
code = strrep(code, "rng('shuffle');", sprintf("rng(%d, 'twister');",seed));
eval(code);
radii = Rs;
end

function radii = packaged_radii(code, scratch, count, branch, seed)
workflow_random_seed = seed; %#ok<NASGU>
workflow_geometry_options = struct('N',count,'command',branch,'mu',2,'sigma',.35); %#ok<NASGU>
code = strrep(code, "fullfile(pwd, 'temp_figs')", "fullfile('"+string(scratch)+"', 'temp_figs')");
eval(code);
radii = Rs;
end
