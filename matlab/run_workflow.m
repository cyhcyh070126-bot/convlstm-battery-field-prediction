function result_dir = run_workflow(output_dir, comsol_mli_dir, c_rate, random_seed, geometry_options)
%RUN_WORKFLOW Generate one microstructure, solve it, and export image frames.
%   RESULT_DIR = RUN_WORKFLOW(OUTPUT_DIR) uses the original 5C setup and a
%   randomly seeded geometry. Run in a MATLAB session connected to COMSOL.
%   Optional arguments: COMSOL_MLI_DIR, C_RATE, RANDOM_SEED, GEOMETRY_OPTIONS.
%   GEOMETRY_OPTIONS may set original N, command (1/2/3), mu, and sigma.
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
if nargin < 5 || isempty(geometry_options)
    geometry_options = struct();
end
geometry_options = validate_geometry_options(geometry_options);
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
fprintf('Checking the connected COMSOL LiveLink session...\n');
try
    com.comsol.model.util.ModelUtil.tags();
catch connection_error
    error('BatteryWorkflow:LiveLinkConnection', ...
        ['Cannot access a connected COMSOL session. Start COMSOL with MATLAB ' ...
         'or connect LiveLink before running this workflow. Adding the mli ' ...
         'folder alone does not establish a connection. No geometry has been ' ...
         'generated. COMSOL reported: %s'], connection_error.message);
end
for required_function = {'normrnd', 'randsample', 'imresize'}
    assert(exist(required_function{1}, 'file') ~= 0, ...
        'Required MATLAB function is unavailable: %s', required_function{1});
end

if ~isfolder(output_dir)
    mkdir(output_dir);
end
[ok, attributes] = fileattrib(output_dir);
assert(ok, 'Could not resolve the output directory.');
result_dir = run_single_simulation(attributes.Name, c_rate, random_seed, geometry_options);
fprintf('Exported simulation: %s\n', result_dir);
end
