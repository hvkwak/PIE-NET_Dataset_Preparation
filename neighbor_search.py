
import tensorflow as tf
DEVICE_BATCH_SIZE = 8

def main():

    # eventually it has to return the tensor...    
    # open_gt_256_64_idx (DEVICE_BATCH_SIZE, 256, 64, dtype = uint16)

    # open_gt_pair_idx (DEVICE_BATCH_SIZE, 256, 64, dtype = uint16) ?
    pred_corner = tf.random.uniform((8, 8096, 2), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32, seed=None, name=None)
    points_cloud = tf.random.uniform((8, 8096, 3), minval=-0.5, maxval=0.5, dtype=tf.dtypes.float32, seed=None, name=None)
    corner_points = tf.where(pred_corner[..., 1] > 0.999)
    batch_corner_pairs_available = [False]*DEVICE_BATCH_SIZE
    batch_open_gt_valid_mask = []

    # organize corner_pairs per batch
    per_batch_corner_pairs = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        idx = tf.boolean_mask(corner_points, corner_points[:,0] == per_batch)[:,1]
        if idx.shape[0] > 1:
            batch_corner_pairs_available[per_batch] = True
            idx_r = tf.repeat(idx, idx.shape[0])
            idx_b = tf.tile(idx, [idx.shape[0]])
            two_col = tf.stack([idx_r, idx_b], 1)
            per_batch_corner_pairs.append(two_col[two_col[:,0] < two_col[:, 1]])
        else:
            per_batch_corner_pairs.append([])
            

    # per batch sample the points
    batch_256_64_idx = []
    batch_open_gt_sample_points = [] # (8, 256, 64, 3)
    corner_pair_open_gt_mask = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        rest_num = 256
        if batch_corner_pairs_available[per_batch]:

            # first increase the precision to float64, 
            # otherwise it may go wrong when it comes to finding points within radius, 
            # where it may also include the two corner points at the end.
            points_cloud = tf.cast(points_cloud, dtype = tf.float64) 

            # find neighbors
            xyz1 = tf.gather(points_cloud[per_batch], indices=per_batch_corner_pairs[per_batch][:, 0], axis=0)
            xyz2 = tf.gather(points_cloud[per_batch], indices=per_batch_corner_pairs[per_batch][:, 1], axis=0)
            ball_center = tf.reduce_mean(tf.stack([xyz1, xyz2], axis = 0), axis = 0)
            distance_from_ball_center = tf.sqrt(tf.reduce_sum(tf.square(tf.math.subtract(tf.expand_dims(ball_center,axis=1), tf.expand_dims(points_cloud[per_batch],axis=0))), axis = 2))
            r = tf.sqrt(tf.reduce_sum(tf.square(xyz1 - xyz2), axis = 1)) / 2.0 # radius
            within_range = tf.math.less(distance_from_ball_center, tf.multiply(tf.ones_like(distance_from_ball_center), tf.expand_dims(r, axis = 1)))

            # per corner pair within this batch, subsample the indicies
            idx_256_64 = []
            corner_pair_num = within_range.shape[0]
            
            # if there are more than 256 pairs, just take first 256.
            if corner_pair_num > 256 : corner_pair_num = 256
            rest_num = 256 - corner_pair_num

            for per_corner in tf.range(corner_pair_num):
                # make sure that corner points(end points) are not within the range.
                assert tf.gather(within_range[per_corner, :], per_batch_corner_pairs[per_batch][per_corner])[0] == tf.constant([False])
                assert tf.gather(within_range[per_corner, :], per_batch_corner_pairs[per_batch][per_corner])[1] == tf.constant([False])
                candidnate_num = tf.where(within_range[per_corner, :]).shape[0]
                #
                # raise error or debug when within_range.shape[0] = 0
                # or if within_range.shape[0] = 3?
                if 64 <= candidnate_num:
                    idx_nums = tf.concat([tf.expand_dims(per_batch_corner_pairs[per_batch][per_corner][0], axis = 0), tf.squeeze(tf.random.shuffle(tf.where(within_range[per_corner, :]))[:62]), tf.expand_dims(per_batch_corner_pairs[per_batch][per_corner][-1], axis = 0)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))

                elif 0 < candidnate_num < 64:
                    n = candidnate_num
                    dummy_num = 64 - 1 - n
                    middle_indicies = tf.squeeze(tf.where(within_range[per_corner, :]))
                    if candidnate_num == 1: 
                        middle_indicies = tf.expand_dims(middle_indicies, axis = 0)
                    idx_nums = tf.concat([tf.expand_dims(per_batch_corner_pairs[per_batch][per_corner][0], axis = 0), middle_indicies, tf.repeat(per_batch_corner_pairs[per_batch][per_corner][-1], dummy_num)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))

            if rest_num > 0: idx_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            batch_256_64_idx.append(tf.concat(idx_256_64, axis = 0))
            batch_open_gt_sample_points.append(tf.gather(points_cloud[per_batch], indices=batch_256_64_idx[per_batch], axis=0))
        else:
            batch_256_64_idx.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            batch_open_gt_sample_points.append(tf.zeros((rest_num, 64, 3), dtype = tf.float32))

        open_gt_valid_mask = tf.expand_dims(tf.cast(tf.sequence_mask(256 - rest_num, 256), dtype=tf.uint8), axis = 1)
        batch_open_gt_valid_mask.append(open_gt_valid_mask)
        # later use tf.gather(points_cloud[0], indices=batch_256_64[0], axis=0) to access the point cloud.

    # (8, 256, 64, 3)
    #corner_pair_open_gt_mask = tf.concat([corner_pair_open_gt_mask[i] for i in range(len(corner_pair_open_gt_mask))], axis = 0)
    corner_pair_sample_points = tf.concat([batch_open_gt_sample_points[i] for i in range(len(batch_open_gt_sample_points))], axis = 0)
    corner_pair_256_64_idx = tf.concat([batch_256_64_idx[i] for i in range(len(batch_256_64_idx))], axis = 0)
    corner_pair_valid_mask = tf.concat([batch_open_gt_valid_mask[i] for i in range(len(batch_open_gt_valid_mask))], axis = 0)
    corner_pair_available = batch_corner_pairs_available

    #batch_open_gt_sample_points = tf.concat([tf.gather(batch_open_gt_sample_points[i], indices = tf.where(batch_open_gt_valid_mask[i])[:, 0], axis = 0) for i in range(len(batch_open_gt_sample_points))], axis = 0)
    return corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_valid_mask, corner_pair_available

if __name__ == "__main__": 
    main()