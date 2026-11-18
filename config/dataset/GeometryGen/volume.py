# Defaults: ~/zoujunhong/data/GeometryGen/volume.
# Override with GEOMETRYGEN_ROOT or GEOMETRYGEN_JOINT_CACHE_ROOT.
# All three batches are cached; index.json records current quality exclusions.
CONFIG={
    'name':'GeometryGenSurfaceVolume',
    'kwargs':{'split':'all','num_surface_points':8192,'num_surface_query_points':2048,'num_query_points':8192,
              'surface_input_list':['xyz','normal'],'surface_target_list':['U','p','surface_distance'],
              'volume_target_list':['U','p','surface_distance'],'surface_distance_target_transform':'sqrt'},
}
