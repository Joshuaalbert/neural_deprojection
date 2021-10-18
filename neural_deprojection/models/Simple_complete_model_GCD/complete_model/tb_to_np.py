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

# model_name = 'discrete_image_vae'
# model_name = 'discrete_voxel_vae'
# model_name = 'auto_regressive_prior'

img_log_dir_path_train = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|disc_image_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32||lrnngrt=4.0e-04,opttyp=adam|||/train/events.out.tfevents.1627465153.node859.35695.2440.v2'
img_log_dir_path_test = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|disc_image_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32||lrnngrt=4.0e-04,opttyp=adam|||/test/events.out.tfevents.1627465153.node859.35695.2448.v2'

vox_log_dir_path_train = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|disc_voxel_vae||embddngdm=32,hddnsz=32,nmchnnls=2,nmembddng=32,nmgrps=4,vxlsprdmnsn=64||lrnngrt=5.0e-04,opttyp=adam|||/train/events.out.tfevents.1627466723.node860.2408.13346.v2'
vox_log_dir_path_test = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|disc_voxel_vae||embddngdm=32,hddnsz=32,nmchnnls=2,nmembddng=32,nmgrps=4,vxlsprdmnsn=64||lrnngrt=5.0e-04,opttyp=adam|||/test/events.out.tfevents.1627466723.node860.2408.13354.v2'

arp_log_dir_path_train = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|auto_regressive_prior||nmhds=4,nmlyrs=2||lrnngrt=4.0e-04,opttyp=adam|||/train/events.out.tfevents.1627635215.node859.49874.25624.v2'
arp_log_dir_path_test = '/home/s2675544/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_GCD/complete_model/log_dir/|auto_regressive_prior||nmhds=4,nmlyrs=2||lrnngrt=4.0e-04,opttyp=adam|||/test/events.out.tfevents.1627635215.node859.49874.25632.v2'

title_names = {'train_one_epoch/while/cond/mini_batch_loss' : 'Minibatch loss',
               'discrete_image_vae/cond/perplexity': 'Perplexity',
               'discrete_image_vae/cond/var_exp': 'Variational expectation',
               'discrete_image_vae/cond/kl_div': 'KL divergence',
               'discrete_image_vae/cond/temperature': 'Temperature',
               'discrete_image_vae/cond/beta': 'Beta',
               'discrete_image_vae/cond/image_predict[0]': 'Input image',
               'discrete_image_vae/cond/image_actual[0]': 'Reconstructed image',
               'discrete_image_vae/cond/latent_logits': 'Logits',
               'discrete_image_vae/cond/latent_samples_onehot': 'Sampled weights',
               'discrete_image_vae/train_one_epoch/epoch_loss': 'Loss',
               'discrete_voxels_vae/train_one_epoch/epoch_loss': 'Loss',
               'auto_regressive_prior/train_one_epoch/epoch_loss': 'Loss',
               'discrete_voxels_vae/cond/perplexity': 'Perplexity',
               'discrete_voxels_vae/cond/var_exp': 'Variational expectation',
               'discrete_voxels_vae/cond/kl_div': 'KL divergence',
               'discrete_voxels_vae/cond/temperature': 'Temperature',
               'discrete_voxels_vae/cond/beta': 'Beta',
               'auto_regressive_prior/perplexity_2d_prior': 'Perplexity 2D',
               'auto_regressive_prior/perplexity_3d_prior': 'Perplexity 3D',
               'auto_regressive_prior/var_exp': 'Variational expectation',
               'auto_regressive_prior/var_exp_2d': 'Variational expectation 2D',
               'auto_regressive_prior/var_exp_3d': 'Variational expectation 3D',
               'auto_regressive_prior/kl_div': 'KL divergence',
               'auto_regressive_prior/kl_div_2d': 'KL divergence 2D',
               'auto_regressive_prior/kl_div_3d': 'KL divergence 3D',
               }

img_ea_train = event_accumulator.EventAccumulator(img_log_dir_path_train,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})
img_ea_test = event_accumulator.EventAccumulator(img_log_dir_path_test,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})

vox_ea_train = event_accumulator.EventAccumulator(vox_log_dir_path_train,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})
vox_ea_test = event_accumulator.EventAccumulator(vox_log_dir_path_test,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})

arp_ea_train = event_accumulator.EventAccumulator(arp_log_dir_path_train,
                                        size_guidance={event_accumulator.TENSORS: number_of_points})
arp_ea_test = event_accumulator.EventAccumulator(arp_log_dir_path_test,
                                                 size_guidance={event_accumulator.TENSORS: number_of_points})

img_ea_train.Reload()
img_ea_test.Reload()
print(img_ea_train.Tags())
print(img_ea_test.Tags())

vox_ea_train.Reload()
vox_ea_test.Reload()
print(vox_ea_train.Tags())
print(vox_ea_test.Tags())

arp_ea_train.Reload()
arp_ea_test.Reload()
print(arp_ea_train.Tags())
print(arp_ea_test.Tags())


def tensor_protos_to_array(df_protos):
    len_array = len(df_protos)
    arr = np.zeros(len_array)

    for i in range(len_array):
        arr[i] = tf.make_ndarray(df_protos[i])

    return arr


def plot_from_dfs(train_df, test_df, name, epoch_or_minibatch='minibatch'):
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
    plt.ylabel(title_names[name])
    plt.title(f'{title_names[name]}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'images/{name.split("/")[0]}_{name.split("/")[-1]}_per_{epoch_or_minibatch}.pdf')
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


df_dict_train = dict()
df_dict_test = dict()

for name in img_ea_train.Tags()['tensors']:
    try:
        if name == 'train_one_epoch/epoch_loss':
            df_dict_train[name] = pd.DataFrame(img_ea_train.Tensors(name))
            df_dict_test['train_one_epoch/loss'] = pd.DataFrame(img_ea_test.Tensors('train_one_epoch/loss'))
            plot_from_dfs(df_dict_train[name], df_dict_test['train_one_epoch/loss'], 'discrete_image_vae/' + name, epoch_or_minibatch='epoch')
        else:
            df_dict_train[name] = pd.DataFrame(img_ea_train.Tensors(name))
            df_dict_test[name] = pd.DataFrame(img_ea_test.Tensors(name))
            plot_from_dfs(df_dict_train[name], df_dict_test[name], name)
        print(name)
    except:
        continue

df_dict_train = dict()
df_dict_test = dict()

for name in vox_ea_train.Tags()['tensors']:
    try:
        if name == 'train_one_epoch/epoch_loss':
            df_dict_train[name] = pd.DataFrame(vox_ea_train.Tensors(name))
            df_dict_test['train_one_epoch/loss'] = pd.DataFrame(vox_ea_test.Tensors('train_one_epoch/loss'))
            plot_from_dfs(df_dict_train[name], df_dict_test['train_one_epoch/loss'], 'discrete_voxels_vae/' + name, epoch_or_minibatch='epoch')
        else:
            df_dict_train[name] = pd.DataFrame(vox_ea_train.Tensors(name))
            df_dict_test[name] = pd.DataFrame(vox_ea_test.Tensors(name))
            plot_from_dfs(df_dict_train[name], df_dict_test[name], name)
        print(name)
    except:
        continue

df_dict_train = dict()
df_dict_test = dict()

for name in arp_ea_train.Tags()['tensors']:
    try:
        if name == 'train_one_epoch/epoch_loss':
            df_dict_train[name] = pd.DataFrame(arp_ea_train.Tensors(name))
            df_dict_test['train_one_epoch/loss'] = pd.DataFrame(arp_ea_test.Tensors('train_one_epoch/loss'))
            plot_from_dfs(df_dict_train[name], df_dict_test['train_one_epoch/loss'], 'auto_regressive_prior/' + name, epoch_or_minibatch='epoch')
        else:
            df_dict_train[name] = pd.DataFrame(arp_ea_train.Tensors(name))
            plot_from_dfs(df_dict_train[name], None, name)
        print(name)
    except:
        continue

# df_train_mini_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# # df_test_mini_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# df_train_epoch_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/epoch_loss'))
# df_test_epoch_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/loss'))
# df_train_var_exp = pd.DataFrame(ea_train.Tensors(model_name + '/cond/var_exp'))
# # df_test_var_exp = pd.DataFrame(ea_test.Tensors(model_name + '/cond/var_exp'))
# df_train_kl_div = pd.DataFrame(ea_train.Tensors(model_name + '/cond/kl_div'))
# # df_test_kl_div = pd.DataFrame(ea_test.Tensors(model_name + '/cond/kl_div'))
# df_train_perplexity = pd.DataFrame(ea_train.Tensors(model_name + '/cond/perplexity'))
# df_test_perplexity = pd.DataFrame(ea_test.Tensors(model_name + '/cond/perplexity'))
# df_train_std_before = pd.DataFrame(ea_train.Tensors(model_name + '/cond_1/properties3_std_before'))
# df_test_std_before = pd.DataFrame(ea_test.Tensors(model_name + '/cond_1/properties3_std_before'))
# df_train_std_after = pd.DataFrame(ea_train.Tensors(model_name + '/cond_1/properties3_std_after'))
# df_test_std_after = pd.DataFrame(ea_test.Tensors(model_name + '/cond_1/properties3_std_after'))

# plot_from_dfs(df_train_mini_loss, None, 'loss')
# plot_from_dfs(df_train_epoch_loss, df_test_epoch_loss, 'loss', 'epoch')
#
# plot_from_dfs(df_train_var_exp, df_train_var_exp, 'variational expectation\n')
# plot_from_dfs(df_train_kl_div, df_train_kl_div, 'kl divergence')
# plot_from_dfs(df_train_perplexity, df_train_perplexity, 'perplexity')
# plot_from_dfs(df_train_std_before, df_train_std_before, 'std before')
# plot_from_dfs(df_train_std_after, df_train_std_after, 'std after')

# plot_from_dfs(df_train_var_exp, df_test_var_exp, 'variational expectation\n')
# plot_from_dfs(df_train_kl_div, df_test_kl_div, 'kl divergence')
# plot_from_dfs(df_train_perplexity, df_test_perplexity, 'perplexity')
# plot_from_dfs(df_train_std_before, df_test_std_before, 'std before')
# plot_from_dfs(df_train_std_after, df_test_std_after, 'std after')
