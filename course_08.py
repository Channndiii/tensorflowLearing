import tensorflow as tf

input_1 = tf.placeholder(tf.float32, shape = [1, 2])
input_2 = tf.placeholder(tf.float32, shape = [2, 1])

output = tf.matmul(input_1, input_2)

with tf.Session() as sess:
    print (sess.run(output,
                    feed_dict={input_1:[[1, 2]],
                               input_2:[[2], [1]]
                              }
                    )
           )