
import tensorflow as tf
import numpy as np
import scipy.io as sio
DEVICE_BATCH_SIZE = 8



def main():

    # eventually it has to return the tensor...    
    # open_gt_256_64_idx (DEVICE_BATCH_SIZE, 256, 64, dtype = uint16)

    # open_gt_pair_idx (DEVICE_BATCH_SIZE, 256, 64, dtype = uint16) ?
    pred_corner = tf.random.uniform((8, 8096, 2), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32, seed=None, name=None)
    points_cloud = tf.random.uniform((8, 8096, 3), minval=-0.5, maxval=0.5, dtype=tf.dtypes.float32, seed=None, name=None)

    corner_points = tf.where(pred_corner[..., 1] > 0.999)
    corner_pair_available = [False]*DEVICE_BATCH_SIZE
    corner_valid_mask_pair = []

    # organize corner_pairs per batch
    corner_pair_idx = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        idx = tf.boolean_mask(corner_points, corner_points[:,0] == per_batch)[:,1]
        if idx.shape[0] > 1:
            corner_pair_available[per_batch] = True
            idx_r = tf.repeat(idx, idx.shape[0])
            idx_b = tf.tile(idx, [idx.shape[0]])
            two_col = tf.stack([idx_r, idx_b], 1)
            corner_pair_idx.append(two_col[two_col[:,0] < two_col[:, 1]])
        else:
            corner_pair_idx.append([])
            

    # per batch sample the points
    corner_pair_256_64_idx = []
    corner_pair_sample_points = [] # (8, 256, 64, 3)
    corner_valid_mask_256_64 = []
    for per_batch in tf.range(DEVICE_BATCH_SIZE, dtype = tf.int64):
        rest_num = 256
        if corner_pair_available[per_batch]:

            # first increase the precision to float64, 
            # otherwise it may go wrong when it comes to finding points within radius, 
            # where it may also include the two corner points at the end.
            points_cloud = tf.cast(points_cloud, dtype = tf.float64) 

            # find neighbors
            xyz1 = tf.gather(points_cloud[per_batch], indices=corner_pair_idx[per_batch][:, 0], axis=0)
            xyz2 = tf.gather(points_cloud[per_batch], indices=corner_pair_idx[per_batch][:, 1], axis=0)
            ball_center = tf.reduce_mean(tf.stack([xyz1, xyz2], axis = 0), axis = 0)
            distance_from_ball_center = tf.sqrt(tf.reduce_sum(tf.square(tf.math.subtract(tf.expand_dims(ball_center,axis=1), tf.expand_dims(points_cloud[per_batch],axis=0))), axis = 2))
            r = tf.sqrt(tf.reduce_sum(tf.square(xyz1 - xyz2), axis = 1)) / 2.0 # radius
            within_range = tf.math.less(distance_from_ball_center, tf.multiply(tf.ones_like(distance_from_ball_center), tf.expand_dims(r, axis = 1)))

            # per corner pair within this batch, subsample the indicies
            idx_256_64 = []
            valid_mask_256_64 = []
            corner_pair_num = within_range.shape[0]
            
            # if there are more than 256 pairs, just take first 256.
            if corner_pair_num > 256 : corner_pair_num = 256
            rest_num = 256 - corner_pair_num

            for per_corner in tf.range(corner_pair_num):
                # make sure that corner points(end points) are not within the range.
                assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[0] == tf.constant([False])
                assert tf.gather(within_range[per_corner, :], corner_pair_idx[per_batch][per_corner])[1] == tf.constant([False])
                candidnate_num = tf.where(within_range[per_corner, :]).shape[0]
                #
                # raise error or debug when within_range.shape[0] = 0
                # or if within_range.shape[0] = 3?
                if 64 <= candidnate_num:
                    idx_nums = tf.concat([tf.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), tf.squeeze(tf.random.shuffle(tf.where(within_range[per_corner, :]))[:62]), tf.expand_dims(corner_pair_idx[per_batch][per_corner][-1], axis = 0)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))
                    valid_mask_256_64.append(tf.expand_dims(tf.ones_like(idx_nums), axis = 0))

                elif 0 < candidnate_num < 64:
                    n = candidnate_num
                    dummy_num = 64 - 1 - n
                    middle_indicies = tf.squeeze(tf.where(within_range[per_corner, :]))
                    if candidnate_num == 1: 
                        middle_indicies = tf.expand_dims(middle_indicies, axis = 0)
                    idx_nums = tf.concat([tf.expand_dims(corner_pair_idx[per_batch][per_corner][0], axis = 0), middle_indicies, tf.repeat(corner_pair_idx[per_batch][per_corner][-1], dummy_num)], axis = 0)
                    idx_256_64.append(tf.expand_dims(idx_nums, axis = 0))
                    valid_mask_256_64.append(tf.expand_dims(tf.concat([tf.ones((64 - (dummy_num - 1)), dtype = tf.int64), tf.zeros((dummy_num - 1), dtype = tf.int64)], axis = 0), axis = 0))

            if rest_num > 0: 
                idx_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
                valid_mask_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_pair_256_64_idx.append(tf.concat(idx_256_64, axis = 0))
            corner_valid_mask_256_64.append(tf.concat(valid_mask_256_64, axis = 0))
            corner_pair_sample_points.append(tf.gather(points_cloud[per_batch], indices=corner_pair_256_64_idx[per_batch], axis=0))
        else:
            corner_pair_256_64_idx.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_valid_mask_256_64.append(tf.zeros((rest_num, 64), dtype = tf.int64))
            corner_pair_sample_points.append(tf.zeros((rest_num, 64, 3), dtype = tf.float32))

        valid_mask = tf.expand_dims(tf.cast(tf.sequence_mask(256 - rest_num, 256), dtype=tf.uint8), axis = 1)
        corner_valid_mask_pair.append(valid_mask)

    return corner_pair_sample_points, corner_pair_256_64_idx, corner_pair_idx, corner_valid_mask_pair, corner_valid_mask_256_64, corner_pair_available
    sample_points = corner_pair_sample_points
    sample_256_64_idx = corner_pair_256_64_idx
    sample_pair_idx = corner_pair_idx
    sample_valid_mask_pair = corner_valid_mask_pair
    sample_valid_mask_256_64 = corner_valid_mask_256_64
    sample_corner_pair_available = corner_pair_available

    my_mat = sio.loadmat('/home/pro2future/Documents/PIE-NET_Dataset_Preparation/3.mat')
    batch_open_gt_pair_idx = np.zeros((DEVICE_BATCH_SIZE, 256, 2), dtype = np.uint16)
    batch_open_gt_256_64_idx = np.zeros((DEVICE_BATCH_SIZE, 256, 64), dtype = np.uint16)

    for m in range(DEVICE_BATCH_SIZE):
        batch_open_gt_pair_idx[m, ...] = my_mat['Training_data'][m, 0]['open_gt_pair_idx'][0, 0]    
        batch_open_gt_256_64_idx[m, ...] = my_mat['Training_data'][m, 0]['open_gt_256_64_idx'][0, 0]

    labels = corner_pair_label_generator(sample_points, \
                                sample_256_64_idx, \
                                sample_pair_idx,\
                                sample_valid_mask, \
                                sample_corner_pair_available, \
                                points_cloud,\
                                batch_open_gt_pair_idx, \
                                batch_open_gt_256_64_idx)
    
    

    #batch_open_gt_sample_points = tf.concat([tf.gather(batch_open_gt_sample_points[i], indices = tf.where(batch_open_gt_valid_mask[i])[:, 0], axis = 0) for i in range(len(batch_open_gt_sample_points))], axis = 0)
    return sample_points, sample_256_64_idx, sample_pair_idx, sample_valid_mask, sample_corner_pair_available

def corner_pair_label_generator(sample_256_64_idx, \
                                sample_pair_idx, \
                                sample_pair_valid_mask, \
                                sample_corner_pairs_available, \
                                points_cloud, \
                                batch_open_gt_pair_idx, \
                                batch_open_gt_256_64_idx):
                                # add more gt_*

    batch_num = len(sample_corner_pairs_available)
    sample_valid_mask_256_64_labels_for_loss = np.zeros((batch_num, 256, 64), dtype = np.int32) # output should be (batch_num, 256, 64, 2)
    sample_valid_mask_pair_labels_for_loss = np.zeros((batch_num, 256, 1), dtype = np.int16)
    points_cloud_np = points_cloud.numpy()
    dist_threshold = 0.01

    # sample_valid_mask_256_64    
    
    for i in range(batch_num):
        # per batch
        if sample_corner_pairs_available[i]:
            sample_valid_mask_pair_numpy = sample_pair_valid_mask[i].numpy()
            k = 0
            found_in_gt_open_pair = False
            while sample_valid_mask_pair_numpy[k][0] == 1:

                # per curve pair k in one batch
                if sample_pair_idx[i][k].numpy() in batch_open_gt_pair_idx[i, :, :]:
                    found_in_gt_open_pair = True
                    # indices match exactly
                    gt_idx = np.where(sample_pair_idx[i][k].numpy() in batch_open_gt_pair_idx[i, :, :])[0]
                    # gt_idx = np.where(batch_open_gt_pair_idx[i][k].numpy() in my_mat['open_gt_pair_idx'][0, 0])[0][0]
                    # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                    mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                    sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                    sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                elif np.flip(sample_pair_idx[i][k].numpy()) in batch_open_gt_pair_idx[i, :, :]:
                    found_in_gt_open_pair = True
                    gt_idx = np.where(np.flip(sample_pair_idx[i][k].numpy()) in batch_open_gt_pair_idx[i, :, :])[0]
                    # my_mat[0, 0]['open_gt_256_64_idx'][gt_idx, :]
                    mask = np.in1d(sample_pair_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                    # update here labels
                    sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                    sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1

                if not found_in_gt_open_pair:
                # not exact match, but see if there is one nearby.
                    # calculate distances NN.
                    distance = np.sqrt(np.sum((points_cloud_np[i][sample_pair_idx[i][k].numpy(), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        found_in_gt_open_pair = True
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                
                if not found_in_gt_open_pair:
                    distance = np.sqrt(np.sum((points_cloud_np[i][np.flip(sample_pair_idx[i][k].numpy()), :] - points_cloud_np[i][batch_open_gt_pair_idx[i, :, :], :])**2, axis = 2))
                    if (distance < np.array([dist_threshold, dist_threshold])).all(axis = 1).sum() > 0:
                        found_in_gt_open_pair = True
                        gt_indices = np.where((distance < np.array([dist_threshold, dist_threshold])).all(axis = 1))[0]
                        gt_idx = gt_indices[np.argmin(distance[gt_indices, :].mean(axis = 1))]
                        mask = np.in1d(sample_256_64_idx[i][k].numpy(), batch_open_gt_256_64_idx[i, :, :][gt_idx])
                        sample_valid_mask_256_64_labels_for_loss[i, k, :] = mask.astype(np.int32)
                        sample_valid_mask_pair_labels_for_loss[i, k, 0] = 1
                k = k+1

    return sample_valid_mask_256_64_labels_for_loss, sample_valid_mask_pair_labels_for_loss


if __name__ == "__main__": 
    main()