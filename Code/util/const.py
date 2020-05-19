# coding=utf-8
class _Const:
    class ConstError(TypeError):
        pass

    class ConstCaseError(ConstError):
        pass

    def __setattr__(self, name, value):
        # 不能修改已经设定的属性
        # if name in self.__dict__:
        #     raise self.ConstError("Can't change const.{}".format(name))

        # 名字必须都大写
        # if not name.isupper():
        #     raise self.ConstCaseError("const name {} is not all uppercase".format(name))
        self.__dict__[name] = value


const = _Const()

const.version = 'exp1'
const.train_path = '../Temp-File/Data/img_processed.csv'
const.vali_path = '../Temp-File/Data/img_processed.csv'
const.test_path = '../Temp-File/Data/img_processed.csv'
const.fc_path = '../Temp-File/Feature-Layer-Output/feature.csv'
const.test_ckpt_path = 'cache/logs_v3_9/min_model.ckpt-27280'
const.ckpt_path = 'logs_v3_10/model.ckpt-59999'

const.weight_decay = 0.00025
const.fc_weight_decay = 0.00025
const.learning_rate = 0.01
const.continue_train_ckpt = False

const.num_residual_blocks = 2
const.is_localization = True
