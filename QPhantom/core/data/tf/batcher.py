import tensorflow as tf

from contextlib import contextmanager

@contextmanager
def batch_data(data_list, sess, batch_size, shuffle=True, n_threads=2):
    """
    TODO: condition shuffle=False
    """
    with tf.device("/cpu"):

        if isinstance(data_list, dict):
            data_names, data_list = list(zip(*list(data_list.items())))
        else:
            data_names = None
        v_data = [tf.convert_to_tensor(d) for d in data_list]
        outputs = tf.train.shuffle_batch(
            v_data,
            batch_size=batch_size,
            capacity=batch_size * 32,
            num_threads=n_threads,
            seed=5,
            enqueue_many=True,
            min_after_dequeue=batch_size * 8
        )
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        def batch_iter(num_batch, data_dict=None):
            with tf.device("/cpu"):
                if data_dict is None:
                    keys = data_names
                elif data_names is None:
                    keys = None
                else:
                    keys = [data_dict[k] for k in data_names]
                for i in range(num_batch):
                    ans = sess.run(outputs)
                    yield ans if data_names is None else {k: v for k, v in zip(keys, ans)}
        yield batch_iter
        coord.request_stop()
        coord.join(threads)
        coord.clear_stop()
