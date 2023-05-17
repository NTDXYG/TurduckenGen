import datetime
import logging
from PLM import PLM_model
from utils import set_seed

set_seed(1234)
lang = 'Java'

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

model_dict = {
    'TurduckenGen' : './codet5-base',
}

model_type = 'TurduckenGen'

# 初始化模型
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], load_model_path=None,
                  beam_size=10, max_source_length=150, max_target_length=256, lang=lang)

start = datetime.datetime.now()
#
# # 模型训练
model.train(train_filename = 'dataset/'+lang+'/train.csv', train_batch_size = 6, learning_rate = 5e-5,
            num_train_epochs = 50, early_stop = 5, do_eval=True, eval_filename='dataset/'+lang+'/valid.csv',
            eval_batch_size=6, output_dir='valid_output_'+lang+'/'+model_type+'/', do_eval_bleu=True)

end = datetime.datetime.now()
print(end-start)

# 加载微调过后的模型参数
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10, max_source_length=150,
                  max_target_length=256, load_model_path='valid_output_'+lang+'/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')

model.sf_test(batch_size=12, filename='dataset/'+lang+'/test.csv', output_dir='test_output_'+lang+'_sf/'+model_type+'/')
