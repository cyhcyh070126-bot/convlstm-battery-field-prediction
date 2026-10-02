function tests = test_export_validation
% Export-contract tests; no COMSOL installation or simulation is required.
tests = functiontests(localfunctions);
end

function setupOnce(testCase)
repo_dir = fileparts(fileparts(fileparts(mfilename('fullpath'))));
original_path = path;
testCase.addTeardown(@() path(original_path));
addpath(fullfile(repo_dir, 'matlab', 'export'));
testCase.TestData.repo_dir = repo_dir;
end

function setup(testCase)
allowed_root = fullfile(testCase.TestData.repo_dir, 'outputs', 'matlab-tests');
if ~isfolder(allowed_root)
    mkdir(allowed_root);
end
fixture_dir = tempname(allowed_root);
mkdir(fixture_dir);
testCase.addTeardown(@remove_fixture, fixture_dir, allowed_root);
testCase.TestData.fixture_dir = fixture_dir;
rgb = zeros(512, 512, 3, 'uint8');
fields = {'1_Concentration', '2_Stress'};
prefixes = {'concentration_t', 'stress_mises_t'};
for field_index = 1:numel(fields)
    mkdir(fullfile(fixture_dir, fields{field_index}));
    for time_value = 0:100:2400
        filename = sprintf('%s%05.0f.png', prefixes{field_index}, time_value);
        imwrite(rgb, fullfile(fixture_dir, fields{field_index}, filename));
    end
end
mkdir(fullfile(fixture_dir, '3_Voronoi_Geometry'));
imwrite(rgb, fullfile(fixture_dir, '3_Voronoi_Geometry', '05_Voronoi_Theta_Colored.png'));
mkdir(fullfile(fixture_dir, 'C-rate'));
rgb(:, :, 1) = 153;
imwrite(rgb, fullfile(fixture_dir, 'C-rate', '5C.png'));
end

function test_complete_sequence(testCase)
actual = validate_simulation_exports(testCase.TestData.fixture_dir, 0:100:2400, 5);
verifyEqual(testCase, actual, 52);
end

function test_missing_c_rate(testCase)
case_dir = testCase.TestData.fixture_dir;
delete(fullfile(case_dir, 'C-rate', '5C.png'));
verifyError(testCase, @() validate_simulation_exports(case_dir, 0:100:2400, 5), ...
    'BatteryWorkflow:MissingExport');
end

function test_missing_frame(testCase)
case_dir = testCase.TestData.fixture_dir;
delete(fullfile(case_dir, '2_Stress', 'stress_mises_t01300.png'));
verifyError(testCase, @() validate_simulation_exports(case_dir, 0:100:2400, 5), ...
    'BatteryWorkflow:MissingExport');
end

function test_wrong_orientation_dimensions(testCase)
case_dir = testCase.TestData.fixture_dir;
imwrite(zeros(256, 512, 3, 'uint8'), ...
    fullfile(case_dir, '3_Voronoi_Geometry', '05_Voronoi_Theta_Colored.png'));
verifyError(testCase, @() validate_simulation_exports(case_dir, 0:100:2400, 5), ...
    'BatteryWorkflow:InvalidExportFormat');
end

function test_grayscale_frame(testCase)
case_dir = testCase.TestData.fixture_dir;
imwrite(zeros(512, 512, 'uint8'), ...
    fullfile(case_dir, '1_Concentration', 'concentration_t00000.png'));
verifyError(testCase, @() validate_simulation_exports(case_dir, 0:100:2400, 5), ...
    'BatteryWorkflow:InvalidExportFormat');
end

function test_ambiguous_c_rate(testCase)
case_dir = testCase.TestData.fixture_dir;
copyfile(fullfile(case_dir, 'C-rate', '5C.png'), fullfile(case_dir, 'C-rate', '1C.png'));
verifyError(testCase, @() validate_simulation_exports(case_dir, 0:100:2400, 5), ...
    'BatteryWorkflow:AmbiguousCRate');
end

function remove_fixture(fixture_dir, allowed_root)
% Resolve both paths before recursive cleanup; only remove our test directory.
[fixture_ok, fixture_info] = fileattrib(fixture_dir);
[root_ok, root_info] = fileattrib(allowed_root);
assert(fixture_ok && root_ok && ...
    startsWith(fixture_info.Name, [root_info.Name filesep], 'IgnoreCase', ispc), ...
    'Refusing to remove a test fixture outside the allowed output directory.');
rmdir(fixture_info.Name, 's');
end
