import sys
import tensorflow as tf
import data_utils
import lstm_model

FLAGS = tf.flags.FLAGS
#parameters

# data parameters
tf.flags.DEFINE_string("train_file", "./data/seg_data/train_data.txt", "Data path for the train data.")
tf.flags.DEFINE_string("valid_file", "./data/seg_data/val_data.val.txt", "Data path for the valid data.")
tf.flags.DEFINE_string("test_file", "./data/seg_data/test_data.txt", "Data path for the test data.")

tf.flags.DEFINE_string("train_data", "./data/seg_data/train_ids.txt", "Data path for the train data.")
tf.flags.DEFINE_string("valid_data", "./data/seg_data/val_ids.txt", "Data path for the valid data.")
tf.flags.DEFINE_string("test_data", "./data/seg_data/test_ids.txt", "Data path for the test data.")

tf.flags.DEFINE_string("save_embedding_file", "./data/embed/embedding_mat.npz", "Embeddings which contains the word from data")
tf.flags.DEFINE_string("pre_train_file", "./data/embed/vector_word.txt", "pre_training embedding file that downloaded.")
tf.flags.DEFINE_string("vocabulary_file", "./data/vocab.txt", "Words in data")
tf.flags.DEFINE_string("model_path", "./model/bilstm/", "save model")

# Model Hyperparameters
tf.flags.DEFINE_integer("vocab_size", 5000, "Number of words in vocabulary")
tf.flags.DEFINE_integer("embedding_dim", 100, "Dimensionality of word embedding(default: 100)")
tf.flags.DEFINE_integer("hidden_size", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("learning_rate", 0.001, "learning rate (default: 0.001)")
tf.flags.DEFINE_integer("max_sent_len", 600, "The max length of sentence (default: 100)")
tf.flags.DEFINE_integer("label_nums", 10, "Number of categories")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 20, "Number of training epochs (default: 20)")


if __name__ == '__main__':

    # data_utils.preprocess(data_paths=[FLAGS.train_file, FLAGS.valid_file, FLAGS.test_file],
    #                       vocab_path=FLAGS.vocabulary_file,
    #                       pre_train_path=FLAGS.pre_train_file,
    #                       embed_mat_path=FLAGS.save_embedding_file,
    #                       vocab_size=FLAGS.vocab_size)

    print('loading pre-training word embedding...')
    word_embedding = data_utils.load_embedding_mat(FLAGS.save_embedding_file)

    print('building model...')
    model = lstm_model.BiLSTM(FLAGS, word_embedding)

    # print('loading test data...')
    # test_data = data_utils.load_data(FLAGS.test_data, FLAGS.max_sent_len)
    # model.test(test_data)


    if len(sys.argv) != 2 or sys.argv[1] not in ['train', 'test']:
        raise ValueError('usage: python run_cnn.py [train / test]')

    if sys.argv[1] == 'train':
        print('loading train and valid data...')
        train_data = data_utils.load_data(FLAGS.train_data, FLAGS.max_sent_len)
        valid_data = data_utils.load_data(FLAGS.valid_data, FLAGS.max_sent_len)
        model.train(train_data, valid_data)
    else:
        print('loading test data...')
        test_data = data_utils.load_data(FLAGS.test_data, FLAGS.max_sent_len)
        model.test(test_data)

