from bert4keras.backend import keras, set_gelu
from bert4keras.tokenizers import Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from keras.layers import Lambda, Dense
from load_data import table_data_filter
import random
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
set_gelu('tanh')

num_classes = 3
maxlen = 64
batch_size = 32
config_path = './albert_small_zh_google/albert_config.json'
checkpoint_path = './albert_small_zh_google/albert_model.ckpt'
dict_path = './albert_small_zh_google/vocab.txt'


def _getDataAndLabel(data):
    returnList = []
    for d in data:
        if d['star'] == '1' or d['star'] == '2':
            returnList.append((d['comments'], 0))
        elif d['star'] == '4' or d['star'] == '5':
            returnList.append((d['comments'], 1))
        elif d['star'] == '3':
            returnList.append((d['comments'], 2))
    return returnList


def _listSplit(fullList, ratio, shuffle=True):
    num = len(fullList)
    offset = int(num * ratio)
    if shuffle:
        random.shuffle(fullList)
    sublist1 = fullList[:offset]
    sublist2 = fullList[offset:]
    return sublist1, sublist2


data = table_data_filter("./data.csv")
data = _getDataAndLabel(data)
trainData, validationAndTestData = _listSplit(data, 0.8)
validationData, testData = _listSplit(validationAndTestData, 0.5)

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """

    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, (text, label) in self.sample(random):
            token_ids, segment_ids = tokenizer.encode(text, maxlen=maxlen)
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append([label])
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids], batch_labels
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


# 加载预训练模型
bert = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    model='albert',
    return_keras_model=False,
)

output = Lambda(lambda x: x[:, 0], name='CLS-token')(bert.model.output)
output = Dense(
    units=num_classes,
    activation='softmax',
    kernel_initializer=bert.initializer
)(output)

model = keras.models.Model(bert.model.input, output)
model.summary()

# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

model.compile(
    loss='sparse_categorical_crossentropy',
    # optimizer=Adam(1e-5),  # 用足够小的学习率
    optimizer=AdamLR(learning_rate=1e-4, lr_schedule={
        1000: 1,
        2000: 0.1
    }),
    metrics=['accuracy'],
)

# 转换数据集
train_generator = data_generator(trainData, batch_size)
valid_generator = data_generator(validationData, batch_size)
test_generator = data_generator(testData, batch_size)


def evaluate(data):
    total, right = 0., 0.
    for x_true, y_true in data:
        y_pred = model.predict(x_true).argmax(axis=1)
        y_true = y_true[:, 0]
        total += len(y_true)
        right += (y_true == y_pred).sum()
    return right / total


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """

    def __init__(self):
        self.best_val_acc = 0.

    def on_epoch_end(self, epoch, logs=None):
        val_acc = evaluate(valid_generator)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            model.save_weights('best_model.weights')
        test_acc = evaluate(test_generator)
        print(
            u'val_acc: %.5f, best_val_acc: %.5f, test_acc: %.5f\n' %
            (val_acc, self.best_val_acc, test_acc)
        )


if __name__ == '__main__':

    evaluator = Evaluator()

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

    model.load_weights('best_model.weights')
    print(u'final test acc: %05f\n' % (evaluate(test_generator)))

else:

    model.load_weights('best_model.weights')
