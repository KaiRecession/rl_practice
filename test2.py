import tensorflow as tf

labels_spare = [0, 2, 1]
logits = tf.constant(value=[[3, 1, -3], [1, 4, 3], [2, 7, 5]],
                     dtype=tf.float32, shape=[3, 3])
loss_spare = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=labels_spare,
    logits=logits
)
print(loss_spare)

labels_sparse = [0, 2, 1]
logit = tf.constant([[3, 1, -3], [1, 4, 3], [2, 7, 5]], dtype=tf.float32)
softmax_logit = tf.nn.softmax(logit)
print(softmax_logit)
softmax_cross_entropy_logit = -(tf.math.log([softmax_logit[0][0], softmax_logit[1][2], softmax_logit[2][1]]))
print(softmax_cross_entropy_logit)

logits = tf.constant([1, 30, 1], dtype=tf.float32)
policy = tf.nn.softmax(logits)

entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)
print(entropy)
