function result_dir = run_single_simulation(workflow_output_root, workflow_c_rate, workflow_random_seed, workflow_geometry_options)
%RUN_SINGLE_SIMULATION Portable orchestration of the original MATLAB scripts.
% Use run_workflow from matlab/ to configure paths and check prerequisites.
% Scientific parameters and image rendering are retained from the source.
if nargin < 4
    workflow_geometry_options = struct();
end
workflow_geometry_options = validate_geometry_options(workflow_geometry_options);

original_directory = pwd;
directory_cleanup = onCleanup(@() cd(original_directory)); %#ok<NASGU>
work_parent = fullfile(workflow_output_root, 'work');
if ~isfolder(work_parent)
    mkdir(work_parent);
end
work_directory = tempname(work_parent);
mkdir(work_directory);
cd(work_directory);

main_D21origin;
N_val = N;
R0_val = R0;
switch command
    case 1
        dist_name_simple = 'Weibull';
    case 2
        dist_name_simple = 'Lognormal';
    case 3
        dist_name_simple = 'Normal';
    otherwise
        error('Unsupported radius distribution: %g', command);
end

build_full_comsol_model;
assert(exist('model', 'var') == 1, 'COMSOL model construction failed.');

C_RATE_TO_RUN = workflow_c_rate;
workflow_time_values = 0:100:2400;
TLIST_TO_RUN = 'range(0,100,2400)';
model.param('par2').set('c_rate', string(C_RATE_TO_RUN));
model.study('std1').feature('time').set('tlist', TLIST_TO_RUN);
fprintf('Solving at C-rate = %g, t = %s s\n', C_RATE_TO_RUN, TLIST_TO_RUN);
model.study('std1').run();

export_simulation_images;
result_dir = fullfile(base_output_folder, run_folder_name);
% Retain the geometry and configuration with each exported image sequence.
copyfile(fullfile(work_directory, 'geometry_for_comsol.mat'), result_dir);
save(fullfile(result_dir, 'run_metadata.mat'), 'workflow_random_seed', 'workflow_geometry_options', ...
    'C_RATE_TO_RUN', 'workflow_time_values', 'N_val', 'R0_val', ...
    'dist_name_simple', 'mu', 'sigma', 'packing_converged', ...
    'k', 'max_force', 'Overlapi', 'ftol', 'ttol');
end
