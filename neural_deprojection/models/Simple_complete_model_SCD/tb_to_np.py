import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
# import seaborn as sns
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
from packaging import version

number_of_points = 1000
model_name = 'simple_complete_model'
# model_name = 'second_discreteImageVAE'
# model_name = 'discreteImageVAE'

log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/single_voxelised_log_dir/|voxelised_model||cmpnntsz=16,dcdr3dhddnsz=4,edgsz=4,glblsz=16,mlthdotptsz=16,nm=simple_complete_model,nmembddng3d=128,nmhds=2,nmprprts=1,vxlprdmnsn=4||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1625680335.node859.38538.2995.v2'
log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/single_voxelised_log_dir/|voxelised_model||cmpnntsz=16,dcdr3dhddnsz=4,edgsz=4,glblsz=16,mlthdotptsz=16,nm=simple_complete_model,nmembddng3d=128,nmhds=2,nmprprts=1,vxlprdmnsn=4||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1625680335.node859.38538.3003.v2'
# log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/second_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=second_discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1625753233.node859.46347.325.v2'
# log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/second_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=second_discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1625753233.node859.46347.333.v2'
# log_dir_path_train = '/home/s1825216/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_SCD/new_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1624624343.node852.52700.269.v2'
# log_dir_path_test = '/home/s1825216/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_SCD/new_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1624624343.node852.52700.277.v2'

ea_train = event_accumulator.EventAccumulator(log_dir_path_train,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})
ea_test = event_accumulator.EventAccumulator(log_dir_path_test,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})

ea_train.Reload()
ea_test.Reload()
print(ea_train.Tags())
print(ea_test.Tags())


def tensor_protos_to_array(df_protos):
    len_array = len(df_protos)
    arr = np.zeros(len_array)

    for i in range(len_array):
        arr[i] = tf.make_ndarray(df_protos[i])

    return arr


def plot_from_dfs(train_df, test_df, property_name, epoch_or_minibatch='minibatch'):
    train_steps = train_df['step'].to_numpy().astype(dtype=float)
    train_prop = tensor_protos_to_array(train_df['tensor_proto'])

    plt.figure(figsize=(4,3))
    plt.plot(train_steps, train_prop, label='train', color='dodgerblue')

    if test_df is not None:
        test_steps = test_df['step'].to_numpy().astype(dtype=float)
        test_prop = tensor_protos_to_array(test_df['tensor_proto'])
        plt.plot(test_steps, test_prop, label='test', color='darkorange')

    if epoch_or_minibatch != 'minibatch':
        plt.xlabel('epoch')
        plt.ticklabel_format(axis='y', style='sci', scilimits=(8, 9))
    else:
        plt.xlabel('minibatch')
        plt.ticklabel_format(axis='both', style='sci', scilimits=(8, 9))
    plt.ylabel(property_name)
    plt.title(f'{property_name} per {epoch_or_minibatch}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{property_name}_per_{epoch_or_minibatch}.pdf')
    plt.close()

# df_train_mini_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# # df_test_mini_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# df_train_epoch_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/epoch_loss'))
# df_test_epoch_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/loss'))
# df_train_var_exp = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond_1/var_exp'))
# df_test_var_exp = pd.DataFrame(ea_test.Tensors(f'{model_name}/cond_1/var_exp'))
# df_train_kl_div = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond_1/kl_div'))
# df_test_kl_div = pd.DataFrame(ea_test.Tensors(f'{model_name}/cond_1/kl_div'))
# df_train_perplexity = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond_1/perplexity'))
# df_test_perplexity = pd.DataFrame(ea_test.Tensors(f'{model_name}/cond_1/perplexity'))


df_train_mini_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# df_test_mini_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
df_train_epoch_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/epoch_loss'))
df_test_epoch_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/loss'))
df_train_var_exp = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/var_exp'))
df_test_var_exp = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/var_exp'))
df_train_kl_div = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/kl_div'))
df_test_kl_div = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/kl_div'))
df_train_perplexity = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/perplexity'))
df_test_perplexity = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/perplexity'))
df_train_std_before = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond_1/properties3_std_before'))
df_test_std_before = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond_1/properties3_std_before'))
df_train_std_after = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond_1/properties3_std_after'))
df_test_std_after = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond_1/properties3_std_after'))

plot_from_dfs(df_train_mini_loss, None, 'loss')
plot_from_dfs(df_train_epoch_loss, df_test_epoch_loss, 'loss', 'epoch')
plot_from_dfs(df_train_var_exp, df_test_var_exp, 'variational expectation\n')
plot_from_dfs(df_train_kl_div, df_test_kl_div, 'kl divergence')
plot_from_dfs(df_train_perplexity, df_test_perplexity, 'perplexity')
plot_from_dfs(df_train_std_before, df_test_std_before, 'std before')
plot_from_dfs(df_train_std_after, df_test_std_after, 'std after')