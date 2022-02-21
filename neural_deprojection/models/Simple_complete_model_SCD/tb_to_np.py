import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
# import seaborn as sns
from scipy import stats
from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import matplotlib.ticker
from packaging import version

number_of_points = 1000
# model_name = 'simple_complete_model'
# model_name = 'second_discreteImageVAE'
# model_name = 'discreteImageVAE'
# model_name = 'discrete_image_vae'
# model_name = 'discrete_voxels_vae'
model_name = 'auto_regressive_prior'

# log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/save_single_voxelised_log_dir/|voxelised_model||cmpnntsz=16,dcdr3dhddnsz=4,edgsz=4,glblsz=16,mlthdotptsz=16,nm=simple_complete_model,nmembddng3d=1024,nmhds=2,nmprprts=1,vxlprdmnsn=4||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1626191003.node853.46283.2996.v2'
# log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/save_single_voxelised_log_dir/|voxelised_model||cmpnntsz=16,dcdr3dhddnsz=4,edgsz=4,glblsz=16,mlthdotptsz=16,nm=simple_complete_model,nmembddng3d=1024,nmhds=2,nmprprts=1,vxlprdmnsn=4||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1626191003.node853.46283.3004.v2'
# log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/second_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=second_discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1625753233.node859.46347.325.v2'
# log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/second_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=second_discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1625753233.node859.46347.333.v2'
# log_dir_path_train = '/home/s1825216/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_SCD/new_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/train/events.out.tfevents.1624624343.node852.52700.269.v2'
# log_dir_path_test = '/home/s1825216/git/neural_deprojection/neural_deprojection/models/Simple_complete_model_SCD/new_im_16_log_dir/|dis_im_vae||embddngdm=64,hddnsz=64,nm=discreteImageVAE,nmchnnls=1,nmembddng=1024||lrnngrt=1.0e-04,opttyp=adam|||/test/events.out.tfevents.1624624343.node852.52700.277.v2'

# log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/log_dir/|disc_image_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32||lrnngrt=4.0e-04,opttyp=adam|||/train/events.out.tfevents.1627581724.node860.603.1895.v2'
# log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/log_dir/|disc_image_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32||lrnngrt=4.0e-04,opttyp=adam|||/test/events.out.tfevents.1627581724.node860.603.1903.v2'

# log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/log_dir/|disc_voxel_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32,vxlsprdmnsn=64||lrnngrt=5.0e-04,opttyp=adam|||/train/events.out.tfevents.1627598304.node860.603.45350.v2'
# log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/log_dir/|disc_voxel_vae||embddngdm=32,hddnsz=32,nmchnnls=1,nmembddng=32,vxlsprdmnsn=64||lrnngrt=5.0e-04,opttyp=adam|||/test/events.out.tfevents.1627598304.node860.603.45358.v2'

log_dir_path_train = '/home/s1825216/Simple_complete_model_SCD/log_dir/|auto_regressive_prior||nmhds=4,nmlyrs=2||lrnngrt=4.0e-04,opttyp=adam|||/train/events.out.tfevents.1627649116.node860.4988.23063.v2'
log_dir_path_test = '/home/s1825216/Simple_complete_model_SCD/log_dir/|auto_regressive_prior||nmhds=4,nmlyrs=2||lrnngrt=4.0e-04,opttyp=adam|||/test/events.out.tfevents.1627649116.node860.4988.23071.v2'

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


def spreads_from_scatter_points(x,y):
    spreads = []
    positions = []

    i = 0
    while i < len(x):
        inds = np.where(x==x[i])[0]
        spread = y[inds]
        spreads.append(spread)
        positions.append(x[i])

        i = inds[-1] + 1 # next set

    return spreads, positions

def plot_from_dfs(train_df, test_df, property_name, epoch_or_minibatch='minibatch'):
    train_steps = train_df['step'].to_numpy().astype(dtype=float)
    train_prop = tensor_protos_to_array(train_df['tensor_proto'])

    test_color = 'dodgerblue'
    test_line_color = 'darkblue'
    train_color = 'orange'

    plt.figure(figsize=(9,3))


    plt.plot(train_steps, train_prop, label='train', color=train_color)

    mf_x = matplotlib.ticker.ScalarFormatter(useMathText=True)
    mf_y = matplotlib.ticker.ScalarFormatter(useMathText=True)

    handles, labels = plt.gca().get_legend_handles_labels()

    if epoch_or_minibatch != 'minibatch':
        plt.xlabel('epoch')
        test_steps = test_df['step'].to_numpy().astype(dtype=float)
        test_prop = tensor_protos_to_array(test_df['tensor_proto'])
        plt.plot(test_steps, test_prop, label='test', color=test_color)
        line = Line2D([0], [0], label='test', color=test_color)
        handles.append(line)
    else:
        plt.xlabel('minibatch')
        if test_df is not None:
            test_steps = test_df['step'].to_numpy().astype(dtype=float)
            test_prop = tensor_protos_to_array(test_df['tensor_proto'])
            # plt.scatter(test_steps, test_prop, label='test', color='darkorange', s=5, zorder=10)

            spreads, positions = spreads_from_scatter_points(test_steps, test_prop)
            plt.boxplot(spreads, positions=positions, widths=0.2*positions[0], whis='range', manage_ticks=False, patch_artist=True,
                        boxprops=dict(facecolor=test_color, color=test_line_color),
                        whiskerprops=dict(color=test_line_color),
                        flierprops=dict(color=test_line_color, markeredgecolor=test_color),
                        capprops=dict(color=test_line_color),
                        medianprops=dict(color=test_line_color),
                        )

            patch = mpatches.Patch(color=test_color, label='test')
            handles.append(patch)

            # plt.plot(mean_steps, means, label=',mean test', color='darkorange', zorder=15, linestyle='--')

    plt.ylabel(property_name)
    mf_x.set_powerlimits((-2, 2))
    mf_y.set_powerlimits((-2, 2))
    plt.gca().yaxis.set_major_formatter(mf_y)
    plt.gca().xaxis.set_major_formatter(mf_x)
    plt.title(f'{property_name} per {epoch_or_minibatch}')
    plt.legend(handles=handles, bbox_to_anchor=(1.01, 1.0), loc='upper left')
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
# df_train_var_exp = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond/var_exp'))
# df_test_var_exp = None
df_train_kl_div_2d = pd.DataFrame(ea_train.Tensors(f'{model_name}/kl_div_2d'))
df_test_kl_div = None
df_train_kl_div_3d = pd.DataFrame(ea_train.Tensors(f'{model_name}/kl_div_3d'))
df_test_kl_div = None
# df_train_perplexity = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond/perplexity'))
# df_test_perplexity = None
# df_train_temperature = pd.DataFrame(ea_train.Tensors(f'{model_name}/cond/temperature'))
# df_test_temperature = None

# df_train_mini_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# # df_test_mini_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/while/cond/mini_batch_loss'))
# df_train_epoch_loss = pd.DataFrame(ea_train.Tensors('train_one_epoch/epoch_loss'))
# df_test_epoch_loss = pd.DataFrame(ea_test.Tensors('train_one_epoch/loss'))
# df_train_var_exp = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/var_exp'))
# df_test_var_exp = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/var_exp'))
# df_train_kl_div = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/kl_div'))
# df_test_kl_div = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/kl_div'))
# df_train_perplexity = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond/perplexity'))
# df_test_perplexity = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond/perplexity'))
# df_train_std_before = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond_1/properties3_std_before'))
# df_test_std_before = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond_1/properties3_std_before'))
# df_train_std_after = pd.DataFrame(ea_train.Tensors('simple_complete_model/cond_1/properties3_std_after'))
# df_test_std_after = pd.DataFrame(ea_test.Tensors('simple_complete_model/cond_1/properties3_std_after'))

plot_from_dfs(df_train_mini_loss, None, 'loss')
plot_from_dfs(df_train_epoch_loss, df_test_epoch_loss, 'loss', 'epoch')
# plot_from_dfs(df_train_var_exp, df_test_var_exp, 'variational expectation\n')
plot_from_dfs(df_train_kl_div_2d, df_test_kl_div, 'kl divergence 2D')
plot_from_dfs(df_train_kl_div_3d, df_test_kl_div, 'kl divergence 3D')
# plot_from_dfs(df_train_perplexity, df_test_perplexity, 'perplexity')
# plot_from_dfs(df_train_temperature, df_test_temperature, 'temperature')
# plot_from_dfs(df_train_std_before, df_test_std_before, 'std before')
# plot_from_dfs(df_train_std_after, df_test_std_after, 'std after')