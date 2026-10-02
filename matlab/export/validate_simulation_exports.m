function file_count = validate_simulation_exports(case_dir, time_values, c_rate)
%VALIDATE_SIMULATION_EXPORTS Check the image contract used by the Python loader.
%   Checks every expected time, both static inputs, and 512-by-512 uint8 RGB
%   decoding. Does not validate COMSOL physics or alter any exported image.

validateattributes(time_values, {'numeric'}, ...
    {'vector', 'nonempty', 'real', 'finite', 'integer', 'nonnegative'});
assert(numel(unique(time_values)) == numel(time_values), ...
    'BatteryWorkflow:DuplicateTime', 'Expected simulation times must be unique.');
file_count = 0;
field_folders = {'1_Concentration', '2_Stress'};
field_prefixes = {'concentration_t', 'stress_mises_t'};
for field_index = 1:numel(field_folders)
    field_dir = fullfile(case_dir, field_folders{field_index});
    for time_value = reshape(time_values, 1, [])
        filename = sprintf('%s%05.0f.png', field_prefixes{field_index}, time_value);
        check_rgb_image(fullfile(field_dir, filename));
        file_count = file_count + 1;
    end
    frames = dir(fullfile(field_dir, '*.png'));
    assert(numel(frames) == numel(time_values), ...
        'BatteryWorkflow:UnexpectedFrames', ...
        'Expected exactly %d PNG frames in %s; found %d.', ...
        numel(time_values), field_dir, numel(frames));
end

check_rgb_image(fullfile(case_dir, '3_Voronoi_Geometry', ...
    '05_Voronoi_Theta_Colored.png'));
crate_dir = fullfile(case_dir, 'C-rate');
check_rgb_image(fullfile(crate_dir, sprintf('%gC.png', c_rate)));
crate_images = dir(fullfile(crate_dir, '*.png'));
assert(numel(crate_images) == 1, 'BatteryWorkflow:AmbiguousCRate', ...
    'Expected exactly one C-rate PNG in %s; found %d.', ...
    crate_dir, numel(crate_images));
file_count = file_count + 2;
fprintf('Validated %d required 512-by-512 RGB images in %s\n', ...
    file_count, case_dir);
end

function check_rgb_image(filename)
assert(isfile(filename), 'BatteryWorkflow:MissingExport', ...
    'A required simulation image is missing: %s', filename);
try
    [pixels, color_map] = imread(filename);
catch image_error
    error('BatteryWorkflow:UnreadableExport', ...
        'Could not decode required image %s: %s', filename, image_error.message);
end
assert(isempty(color_map) && isa(pixels, 'uint8') && ...
    isequal(size(pixels), [512, 512, 3]), ...
    'BatteryWorkflow:InvalidExportFormat', ...
    'Expected a 512-by-512 uint8 RGB image, without an indexed palette: %s', filename);
end
