function result_dir = run_workflow(output_dir, comsol_mli_dir, c_rate, random_seed)
%RUN_WORKFLOW Generate one microstructure, solve it, and export image frames.
%   RESULT_DIR = RUN_WORKFLOW(OUTPUT_DIR) uses the original 5C setup and a
%   randomly seeded geometry. Run in a MATLAB session connected to COMSOL.
%   Optional arguments: COMSOL_MLI_DIR, C_RATE, RANDOM_SEED.
%   See docs/matlab-workflow.md for setup, defaults, and limitations.

matlab_dir = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(fileparts(matlab_dir), 'outputs', 'matlab');
end
if nargin < 2
    comsol_mli_dir = '';
end
if nargin < 3 || isempty(c_rate)
    c_rate = 5;
end
if nargin < 4
    random_seed = [];
end
validateattributes(c_rate, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
if ~isempty(random_seed)
    validateattributes(random_seed, {'numeric'}, ...
        {'scalar', 'integer', 'nonnegative', '<=', 2^32 - 1});
end

original_path = path;
path_cleanup = onCleanup(@() path(original_path)); %#ok<NASGU>
addpath(fullfile(matlab_dir, 'geometry'));
addpath(fullfile(matlab_dir, 'simulation'));
addpath(fullfile(matlab_dir, 'export'));
if ~isempty(comsol_mli_dir)
    assert(isfolder(comsol_mli_dir), 'COMSOL mli directory does not exist.');
    addpath(comsol_mli_dir);
end
assert(exist('mphsave', 'file') ~= 0, ...
    ['COMSOL LiveLink functions are unavailable. Start COMSOL with MATLAB, ' ...
     'or configure a connected LiveLink session and supply its mli directory.']);
for required_function = {'normrnd', 'randsample', 'imresize'}
    assert(exist(required_function{1}, 'file') ~= 0, ...
        'Required MATLAB function is unavailable: %s', required_function{1});
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end
[ok, attributes] = fileattrib(output_dir);
assert(ok, 'Could not resolve the output directory.');
result_dir = run_single_simulation(attributes.Name, c_rate, random_seed);
fprintf('Exported simulation: %s\n', result_dir);
end
