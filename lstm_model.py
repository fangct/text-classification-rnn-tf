import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import classification_report

os.environ["CUDA_VISIBLE_DEVICES"] = "3, 6"

class BiLSTM(object):

    def __init__(self, flags, embedding):
        self.flags = flags
        self.pre_embedding = embedding

        self.build_model()

    def place_holder(self):
        self.input_x = tf.placeholder(dtype=tf.int32, shape=[None, self.flags.max_sent_len], name='input_x')
        self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, self.flags.label_nums], name='input_y')
        self.dropout = tf.placeholder(dtype=tf.float32, shape=[], name='dropout')

    def embedding(self):
        if self.pre_embedding is None:
            self.W_embed = tf.get_variable(name='embedding', shape=[self.flags.vocab_size, self.flags.embedding_dim],
                                               initializer=tf.random_uniform_initializer(-0.1, 0.1), trainable=True)
        else:
            self.W_embed = tf.Variable(self.pre_embedding, dtype=tf.float32, trainable=True, name='word_embedding')

        self.word_embedding = tf.nn.embedding_lookup(self.W_embed, self.input_x) #(?, 600, 100)

    def bilstm(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.flags.hidden_size)
        cell = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell, output_keep_prob=self.flags.dropout_keep_prob)
        output, _ = tf.nn.dynamic_rnn(cell=cell, inputs=self.word_embedding, dtype=tf.float32) # (?, 600, 128)

        self.last = output[:, -1, :]  # 取最后一个时序输出作为结果 (?, 128)

    def fc_layer(self):
        self.W_fc = tf.get_variable(name='w_fc', shape=[self.flags.hidden_size, self.flags.label_nums], initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.b_fc = tf.get_variable(name='b_fc', shape=[self.flags.label_nums], initializer=tf.constant_initializer(0.1))

        self.logits = tf.add(tf.matmul(self.last, self.W_fc), self.b_fc)

        self.correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.input_y, 1))
        self.acc = tf.reduce_mean(tf.cast( self.correct_pred, tf.float32))

    def train_op(self):
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits))

        basic_lr = self.flags.learning_rate
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(basic_lr, global_step=global_step, decay_steps=200, decay_rate=0.99, staircase=True)

        self.train_step = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step)

    def build_model(self):
        self.place_holder()
        self.embedding()
        self.bilstm()
        self.fc_layer()
        self.train_op()

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()

    def batch_iter(self, data, batch_size):
        x, y = data[0], data[1]
        data_size = len(x)
        num_batch = data_size // batch_size + 1
        indices = np.random.permutation(np.arange(data_size))

        x_shuff = x[indices]
        y_shuff = y[indices]

        for i in range(num_batch):
            start = i * batch_size
            end = min((i + 1) * batch_size, data_size)
            yield x_shuff[start:end], y_shuff[start:end]

    def evaluate(self, data):
        print('evaluating...')
        total_acc = 0.0
        total_loss = 0.0
        batch_data = self.batch_iter(data, self.flags.batch_size)
        for batch_x, batch_y in batch_data:
            batch_len = len(batch_x)
            loss, acc = self.sess.run([self.loss, self.acc],
                                      feed_dict={self.input_x: batch_x, self.input_y: batch_y, self.dropout: 1.0})
            total_loss += loss * batch_len
            total_acc += acc * batch_len
        return total_loss / len(data[0]), total_acc / len(data[0])

    def train(self, train_data, val_data):
        print('training...')
        best_acc = 0.0

        log_path = os.path.join(self.flags.model_path, 'output', 'log.txt')
        f = open(log_path, 'a+', encoding='utf-8')

        for e in range(self.flags.num_epochs):
            print('epoch #{}'.format(e))
            batch_data = self.batch_iter(train_data, self.flags.batch_size)
            for i, (x_train, y_train) in enumerate(batch_data):

                _, lr = self.sess.run([self.train_step, self.learning_rate],
                                      feed_dict={self.input_x: x_train, self.input_y: y_train, self.dropout: self.flags.dropout_keep_prob})
                train_loss, train_acc = self.sess.run([self.loss, self.acc],
                                                      feed_dict={self.input_x: x_train, self.input_y: y_train,
                                                                 self.dropout: self.flags.dropout_keep_prob})

                if i% 100 == 0 and i != 0:

                    val_loss, val_acc = self.evaluate(val_data)
                    print('train_loss:{:.4}, train_acc:{:.2%}, val_loss:{:.4}, val_acc:{:.2%}, best_acc:{:.2%}'.format( train_loss, train_acc, val_loss, val_acc, best_acc))
                    f.write('train_loss:{:.4}, train_acc:{:.2%}, val_loss:{:.4}, val_acc:{:.2%}, best_acc:{:.2%} \n'.format( train_loss, train_acc, val_loss, val_acc, best_acc))


                    if val_acc > best_acc:
                        best_acc = val_acc
                        print('Accuracy improvement，save model to {}'.format(self.flags.model_path))
                        self.saver.save(self.sess, self.flags.model_path)

        f.close()


    def test(self, test_data):
        print('testing...')
        print('restore model from {}'.format(self.flags.model_path))
        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, self.flags.model_path)

        x_test, y_test = test_data[0], test_data[1]
        y_true = np.argmax(y_test, 1)
        y_pred = np.zeros(shape=(len(y_test)), dtype=np.int32)

        total_acc = 0.0
        total_loss = 0.0
        step = len(x_test) // self.flags.batch_size + 1
        for i in range(step):
            start = i * self.flags.batch_size
            end = min(start + self.flags.batch_size, len(x_test))
            batch_x, batch_y = x_test[start:end], y_test[start:end]
            loss, acc = sess.run([self.loss, self.acc],
                                 feed_dict={self.input_x: batch_x, self.input_y: batch_y, self.dropout: 1.0})

            total_loss += loss * self.flags.batch_size
            total_acc += acc * self.flags.batch_size

            pred = tf.argmax(self.logits, 1)
            batch_pred = sess.run(pred, feed_dict={self.input_x: batch_x, self.dropout: 1.0})

            y_pred[start:end] = batch_pred

        test_loss = total_loss / len(x_test)
        test_acc = total_acc / len(x_test)
        print('test_loss:{:.4}, test_acc:{:.2%}'.format(test_loss, test_acc))

        # compute P R F1
        categories = ['体育', '财经', '房产', '家居', '教育', '科技', '时尚', '时政', '游戏', '娱乐']
        print(classification_report(y_true, y_pred, target_names=categories))








