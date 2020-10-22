import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import numpy as np
from termcolor import colored
from tabulate import tabulate
import global_parameters as gp


class MyFig:
    def __init__(self, num_of_subfig: list, figsize):
        self.fig = plt.figure(figsize=figsize, tight_layout=True)
        self.ax_list = []
        for index_i in range(1, int(num_of_subfig[0] * num_of_subfig[1] + 1)):
            self.ax_list.append(self.fig.add_subplot(num_of_subfig[0], num_of_subfig[1], index_i))
        self.num_of_subfig = num_of_subfig
        self.im_list = [None for _ in range(0, int(num_of_subfig[0] * num_of_subfig[1]))]
        self.data_list = [np.array([]) for _ in range(0, int(num_of_subfig[0] * num_of_subfig[1]))]
        self.index_h = 0
        self.index_v = 0
        self.index = 0

    def close(self):
        plt.close(self.fig)

    def next_figure(self):
        if self.index_v >= self.num_of_subfig[1] and self.index_h >= self.num_of_subfig[0]:
            raise IndexError("No next figure")
        if self.index_h == self.num_of_subfig[0] - 1:
            self.index_h = 0
            self.index_v += 1
        else:
            self.index_h += 1
        self.index += 1

    def reset_index(self):
        self.index_h = 0
        self.index_v = 0
        self.index = 0

    def plot_grid(self, data: np.ndarray, max_min: (int, int, int, int), step_size: int,
                  range_value: list, title: str):
        # max_min: vmax, vmin, hmax, hmin
        vmax, vmin, hmax, hmin = max_min
        self.data_list[self.index] = data
        data[data == 0] = gp.PLOT_FADING_RANGE_LOG[1]
        norm = mcolors.Normalize(vmin=range_value[0], vmax=range_value[1])
        # see note above: this makes all pcolormesh calls consistent:
        pc_kwargs = {'cmap': 'pink', 'norm': norm, 'edgecolors': 'k', 'linewidths': 2}
        self.ax_list[self.index].set_title(title)
        self.ax_list[self.index].grid(color='k', linestyle='-', linewidth=1)
        self.im_list[self.index] = self.ax_list[self.index].pcolor(self.data_list[self.index],
                                                                   vmin=gp.PLOT_FADING_RANGE_LOG[0],
                                                                   vmax=gp.PLOT_FADING_RANGE_LOG[1],
                                                                   **pc_kwargs)
        plt.setp(self.ax_list[self.index], xticks=np.arange(0, hmax - hmin + 2),
                 xticklabels=np.around(np.arange(hmin * step_size, (hmax + 2) * step_size,
                                                 step=step_size), decimals=1),
                 yticks=np.arange(0, vmax - vmin + 2),
                 yticklabels=np.around(np.arange(vmin * step_size, (vmax + 2) * step_size + step_size,
                                                 step=step_size), decimals=1))
        # +2 because 2-6 has 6 - 2 + 1 = 5 elements, range(2, 7, 1) count 6 (+1)
        # some element has position larger than 6 * step, so we need 7 also (+1)
        self.fig.colorbar(self.im_list[self.index], ax=self.ax_list[self.index], shrink=1, extend='min')

    def get_color(self, val: float, cmap):
        # Return the data color of an index.
        if int(val) > 0:
            raise ValueError("Fading can't larger than 0")
        return cmap(1 - abs(val / np.min(self.data_list[self.index])))

    def draw_text_label(self, data, position: (int, int), idex: int):
        facecolor = self.get_color(data, self.im_list[self.index].get_cmap())
        self.ax_list[self.index].text(position[0], position[1], "<{id}>".format(id=idex),
                                      color='white', ha='center', va='center',
                                      bbox={'boxstyle': 'square', 'facecolor': facecolor})

    def draw_text_block(self, rotation: int, position: (float, float), hori_axis: str, verti_axis: str, content: str):
        # hori_axis/verti_axis = 'left' 'right' 'center'
        bbox_kwargs = {'fc': 'w', 'alpha': .75, 'boxstyle': "round4"}
        ann_kwargs = {'xycoords': 'axes fraction', 'textcoords': 'offset points', 'bbox': bbox_kwargs}
        self.ax_list[self.index].annotate(content, xy=position, xytext=(0, 0),
                                          ha=hori_axis, va=verti_axis, rotation=rotation, **ann_kwargs)

    def save_figure(self, time, apid):
        title_str = "clustering_result at time {ti} for access point {apid}".format(ti=time, apid=apid)
        self.fig.suptitle(title_str, fontsize=16)
        plt.savefig("./fig/cluster_result/" + title_str + ".eps")


def plot_observation(observation: np.ndarray, group_size: int, num_of_ap: int, title: str, save=True):
    fig = plt.figure(figsize=(gp.IMAGE_SIZE[0] * observation.shape[2],
                              gp.IMAGE_SIZE[1] * 1), tight_layout=True)
    ax_list = []
    for index_i in range(observation.shape[2]):
        ax_list.append(fig.add_subplot(1, observation.shape[2], index_i + 1))
    im_list = [None for _ in range(observation.shape[2])]
    data_list = [observation[:, :, _] for _ in range(observation.shape[2])]

    for index, data in enumerate(data_list):
        pc_kwargs = {'cmap': 'binary', 'edgecolors': 'k', 'linewidths': 2}
        ax_list[index].set_title(
            "Observation {a} for group {b}".format(a=index % group_size, b=index // group_size + 1))
        ax_list[index].grid(color='k', linestyle='-', linewidth=1)
        im_list[index] = ax_list[index].pcolor(data_list[index], vmin=0, vmax=1, **pc_kwargs)
        plt.setp(ax_list[index])

        bbox_kwargs = {'fc': 'w', 'alpha': .75, 'boxstyle': "round4"}
        ann_kwargs = {'xycoords': 'axes fraction', 'textcoords': 'offset points', 'bbox': bbox_kwargs}
        ap_len = np.floor(np.sqrt(num_of_ap)).astype(int)
        min_space = (1 / ap_len) / 2
        space = 1 / ap_len
        for ind in range(num_of_ap):
            ax_list[index].annotate(str(ind), xy=(min_space + ind % 2 * space, min_space + ind // 2 * space), xytext=(0, 0),
                                    ha='center', va='center', rotation=0, **ann_kwargs)

    if save:
        fig.suptitle(title, fontsize=16)
        plt.savefig("./fig/decision/" + title + ".eps")


def table_print_color(table: np.ndarray, title: str, color):
    indi_r = np.indices([table.shape[0]])
    print(colored(title, color))
    if table.shape.__len__() == 1:
        if table.dtype == np.complex:
            temp_table = np.zeros((table.shape[0], 1, 2))
            temp_table[:, :, 0] = [np.real(table)]
            temp_table[:, :, 1] = [np.imag(table)]
            print(colored(tabulate([temp_table], headers=[str(k) for k in indi_r[0]], tablefmt="grid"), color))
        else:
            print(colored(tabulate([table], headers=[str(k) for k in indi_r[0]], tablefmt="grid"), color))
    else:
        indi_c = np.indices([table.shape[1]])
        if table.dtype == np.complex:
            temp_table = np.zeros((table.shape[0], table.shape[1], 2))
            temp_table[:, :, 0] = np.real(table)
            temp_table[:, :, 1] = np.imag(table)
            print(colored(tabulate(np.insert(temp_table, 0, np.expand_dims(indi_r, axis=2), axis=1),
                                   headers=['ID'] + [str(k) for k in indi_c[0]], tablefmt="grid"), color))
        else:
            print(colored(tabulate(np.insert(table, 0, indi_r, axis=1),
                                   headers=['ID'] + [str(k) for k in indi_c[0]], tablefmt="grid"), color))


def print_all_nodes_information(target):
    if gp.LOG_LEVEL >= 2:
        result_dic = []
        for ues in target.users_list:
            result_dic.append([ues.id, ues.position, ues.sphere_id, ues.resource, ues.transmission_mask])
        print(colored("UE LIST", gp.UE_COLOR))
        print(colored(tabulate(result_dic, headers=['ID', 'Position', 'Sphere ID', 'Resource', 'Transmission Mask'],
                               tablefmt="grid"), gp.UE_COLOR))

        result_dic = []
        for aps in target.accesspoint_list:
            result_dic.append([aps.id, aps.position])
        print(colored("AP LIST", gp.AP_COLOR))
        print(colored(tabulate(result_dic, headers=['ID', 'Position'], tablefmt="grid"), gp.AP_COLOR))

        result_dic = []
        for uavs in target.uav_list:
            result_dic.append([uavs.id, uavs.position, uavs.resource])
        print(colored("UAV LIST", gp.CS_COLOR))
        print(colored(tabulate(result_dic, headers=['ID', 'Position', 'Resources'], tablefmt="grid"), gp.CS_COLOR))


def print_users_information(target, transmitted_list):
    if gp.LOG_LEVEL >= 2:
        result_dic = []
        for ues in target.users_list:
            if ues.sphere_id in target.stack_of_popular:
                # ue's cluster may not be selected
                if transmitted_list[ues.sphere_id].__len__() != 0:
                    result_dic.append([ues.id, ues.position, ues.sphere_id, transmitted_list[ues.sphere_id],
                                       [res for key, res in enumerate(ues.resource) if ues.transmission_mask[key]]])
            else:
                result_dic.append([ues.id, ues.position, ues.sphere_id, -1,
                                   [res for key, res in enumerate(ues.resource) if ues.transmission_mask[key]]])
        print(colored("UE LIST", gp.AP_COLOR))
        print(colored(tabulate(result_dic, headers=['ID', 'Position', 'Sphere ID', 'Current Transmitted',
                                                    'Remained Resources'],
                               tablefmt="grid"), gp.AP_COLOR))
