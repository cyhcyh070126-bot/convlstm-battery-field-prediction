function options = validate_geometry_options(options)
%VALIDATE_GEOMETRY_OPTIONS Validate optional overrides of original parameters.
% No sampling, packing, physics, or rendering implementation is defined here.
assert(isstruct(options) && isscalar(options), ...
    'Geometry options must be a scalar struct.');
allowed = {'N', 'command', 'mu', 'sigma'};
assert(all(ismember(fieldnames(options), allowed)), ...
    'Supported geometry fields are N, command, mu, and sigma.');
if isfield(options, 'N')
    validateattributes(options.N, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 3});
end
if isfield(options, 'command')
    validateattributes(options.command, {'numeric'}, ...
        {'scalar', 'real', 'finite', 'integer', '>=', 1, '<=', 3});
end
for name = {'mu', 'sigma'}
    if isfield(options, name{1})
        validateattributes(options.(name{1}), {'numeric'}, ...
            {'scalar', 'real', 'finite', 'positive'});
    end
end
end
