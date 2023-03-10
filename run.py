import datetime
import logging
from PLM import PLM_model
from utils import set_seed

set_seed(1234)

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
                  beam_size=10, max_source_length=150, max_target_length=256)

start = datetime.datetime.now()
#
# # 模型训练
model.train(train_filename = 'dataset/python/train.csv', train_batch_size = 6, learning_rate = 5e-5,
            num_train_epochs = 50, early_stop = 5, do_eval=True, eval_filename='dataset/python/valid.csv',
            eval_batch_size=6, output_dir='valid_output_python/'+model_type+'/', do_eval_bleu=True)

end = datetime.datetime.now()
print(end-start)

# 加载微调过后的模型参数
model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=10, max_source_length=150,
                  max_target_length=256, load_model_path='valid_output_python/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin')

model.test(batch_size=12, filename='dataset/python/test.csv', output_dir='test_output_python/'+model_type+'/')

model.sf_test(batch_size=12, filename='dataset/python/test.csv', output_dir='test_output_python_gebs/'+model_type+'/')
# for beam in range(2, 15):
#     model = PLM_model(model_type=model_type, model_name_or_path=model_dict[model_type], beam_size=beam,
#                       max_source_length=150, max_target_length=256, load_model_path='valid_output_pcython/'+model_type+'/checkpoint-best-bleu/pytorch_model.bin', method_type=3)

#     model.test(batch_size=2, filename='dataset/python/test.csv', output_dir='test_output_python/'+str(beam)+model_type+'/')

#     model.greedy_test(batch_size=2, filename='dataset/python/test.csv', output_dir='test_output_python_gebs/'+str(beam)+model_type+'/')

# model.multinomial_sampling(batch_size=12, filename='dataset/python/test.csv', output_dir='test_output_python_multinomial_sampling/'+model_type+'/')

# model.greedy_search_test(batch_size=12, filename='dataset/python/test.csv', output_dir='test_output_python_greedy_search/'+model_type+'/')