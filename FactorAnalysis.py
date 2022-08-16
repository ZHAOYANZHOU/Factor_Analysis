import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import scipy.stats
import concurrent
# 图画面板调整为白色
rc = {'axes.facecolor': 'white',
      'savefig.facecolor': 'white'}
mpl.rcParams.update(rc)
# 显示负号
mpl.rcParams['axes.unicode_minus']=False

# 提高照片清晰度 dpi:dots per inch
mpl.rcParams['figure.dpi'] = 200

# 设置样式
plt.style.use('_mpl-gallery')


class FactorAnalysis:
    def __init__(self, factor, ret, list_periods, n_group):
        self.factor = factor
        self.ret = ret
        self.list_periods = list_periods
        self.n_group = n_group

    # def summary_group_ic(self, group_list):
    #     # 在每一个调仓期间
    #     summary = []
    #     for n_group in group_list:
    #         results = []
    #         adj_date = []
    #         for adj_idx in range(0, len(self.factor), 1):
    #             adj_day = self.factor.index[adj_idx]
    #             adj_date.append(adj_day)
    #             print(adj_day)  # 这是根据因子值调仓日，在开盘的时候，用的open数据
    #
    #             # 获取每个周期的因子值和收益率，并合并
    #             df_1 = self.factor.iloc[adj_idx, :].to_frame().copy()
    #             df_1.columns = ['factor']
    #             df_1 = pd.concat([df_1, self.ret.iloc[adj_idx: adj_idx+1, :].copy().T],
    #                              axis=1)
    #
    #             # 根据factor的值去掉nan，没有用到未来的股价信息
    #             df_1.dropna(subset=['factor'], inplace=True)
    #
    #             # 给factor添加一个很小的随机扰动
    #             df_1['factor'] += np.random.normal(0, 1, size=(df_1.shape[0])) * 1e-18
    #
    #             # 先不考虑分组不够的情况
    #
    #             # ==============================
    #             # 计算该调仓周期的各股收益率
    #             # ==============================
    #             # 如果调仓第一天有nan，说明买不进去，可以砍掉这一条股票，因为交易第二天确实买不进去
    #             df_1.dropna(subset=[df_1.columns[1]], inplace=True)
    #             # 如果调仓第一天之后有nan，保持收益率为0不变，cumsum会砍掉部分收益
    #             df_1.iloc[:, 2:] = df_1.iloc[:, 2:].fillna(0)
    #             df_1['ret'] = np.cumprod(1 + df_1.iloc[:, 1:], axis=1).iloc[:, -1] - 1
    #
    #             df_2 = df_1.loc[:, ['factor', 'ret']].copy()
    #             # 计算每个组的分割线，并fillna对数据进行分组
    #             # 有个小缺点就是最后一组的数据可能会少一些
    #             # df_2.sort_values(by=['factor'], ascending=True, inplace=True)
    #             # df_quantile = df_2['factor'][np.arange(0, len(df_2), int(np.ceil(len(df_2) / n_group)))].to_frame()
    #             # df_quantile['group'] = np.arange(1, n_group+1)
    #             # 根据factor进行分组
    #             df_2['group'] = pd.qcut(df_2['factor'], n_group, labels=list(range(1, n_group+1)))
    #
    #             # 计算分组收益率
    #             # groupby会直接过滤掉ret为nan的股票，也就说你买不进去，符合常理
    #             df_ret = df_2.groupby(by=['group'])['ret'].mean().to_frame()  # 用普通收益率计算的分组收益率
    #             corr = stats.spearmanr(df_ret.index, df_ret['ret'])[0]
    #             results.append(corr)
    #
    #         summary.append(results)
    #     df_summary = pd.DataFrame(data=summary, columns=adj_date,
    #                               index=group_list)
    #     return df_summary

    # ==============================
    # ==============================
    # 计算return，ic，rank_ic，换手率，每组数量
    # ==============================
    # ==============================
    def calc_results(self, adj_periods):
        adj_date = []
        ret_results = pd.DataFrame()
        excess_ret_results = pd.DataFrame()
        ex_results = pd.DataFrame()
        ic_list = []
        rank_ic_list = []
        df_last = pd.DataFrame()
        df_group_num = pd.DataFrame()
        group_ic_list = []
        df_group_inner_ic = pd.DataFrame()

        # 在每一个调仓期间
        # 我也在想要不要写函数，但是发现在每个调仓期间写函数也不方便
        for adj_idx in range(0, len(self.factor), adj_periods):
            adj_day = self.factor.index[adj_idx]
            adj_date.append(adj_day)
            # print(adj_day)  # 这是根据因子值调仓日，在开盘的时候，用的open数据

            # 获取每个周期的因子值和收益率，并合并
            df_1 = self.factor.iloc[adj_idx, :].to_frame()
            df_1.columns = ['factor']
            df_1 = pd.concat([df_1, self.ret.iloc[adj_idx: adj_idx+adj_periods, :].T],
                             axis=1)

            # 根据factor的值去掉nan，没有用到未来的股价信息
            df_1.dropna(subset=['factor'], inplace=True)

            # 给factor添加一个很小的随机扰动
            df_1['factor'] += np.random.normal(0, 1, size=(df_1.shape[0])) * 1e-18

            # 先不考虑分组不够的情况

            # ==============================
            # 计算该调仓周期的各股收益率
            # ==============================
            # 如果调仓第一天有nan，说明买不进去，砍掉。这里没有用到未来数据，因为确实买不进去
            df_1.dropna(subset=[df_1.columns[1]], inplace=True)
            # 如果调仓第一天之后有nan，保持收益率为0不变，cumsum会砍掉部分收益
            df_1.iloc[:, 2:] = df_1.iloc[:, 2:].fillna(0)
            # 计算调仓周期内的收益率
            df_1['ret'] = np.cumprod(1 + df_1.iloc[:, 1:], axis=1).iloc[:, -1] - 1

            # 根据group进行分组收益率的计算
            df_2 = df_1.loc[:, ['factor', 'ret']].copy()
            df_2['group'] = pd.qcut(df_2['factor'], self.n_group, labels=list(range(1, self.n_group+1)))

            # # 计算每组内部的ic情况
            # inner_ic_list = []
            # for group_idx in range(1, self.n_group+1):
            #     inner_ic = df_2[df_2['group'] == group_idx]
            #     inner_ic = scipy.stats.spearmanr(inner_ic.index, inner_ic['ret'])[0]
            #     inner_ic_list.append(inner_ic)
            # df_inner_ic = pd.DataFrame(data=inner_ic_list,
            #                            index=list(range(1, self.n_group+1)),
            #                            columns=[adj_day])
            # df_group_inner_ic = pd.concat([df_group_inner_ic, df_inner_ic], axis=1)

            # 记录一下每一组的个数，方便查看是否异常
            df_group_num_tmp = df_2['group'].value_counts().to_frame()
            df_group_num_tmp.columns = [adj_day]
            df_group_num = pd.concat([df_group_num, df_group_num_tmp], axis=1)

            # 计算分组收益率
            # groupby会直接过滤掉ret为nan的股票，也就说你买不进去，符合常理
            df_ret = df_2.groupby(by=['group'])['ret'].mean().to_frame()  # 用普通收益率计算的分组收益率
            # 顺手计算一下分层的ic
            group_ic = scipy.stats.spearmanr(df_ret.index, df_ret['ret'])[0]
            group_ic_list.append(group_ic)

            df_ret.columns = [adj_day]
            ret_results = pd.concat([ret_results, df_ret], axis=1)

            # # 计算分组的超额收益率
            # market_ret = df_2['ret'].mean()  # 用普通收益率计算的市场收益率
            # df_excess_ret = df_ret - market_ret
            # excess_ret_results = pd.concat([excess_ret_results, df_excess_ret], axis=1)

            # ==============================
            # 计算IC, IC_IR
            # ==============================
            ic = df_1.loc[:, ['factor', 'ret']].corr(method='pearson').iloc[0, 1]
            rank_ic = df_1.loc[:, ['factor', 'ret']].corr(method='spearman').iloc[0, 1]
            ic_list.append(ic)
            rank_ic_list.append(rank_ic)

            # ==============================
            # 计算一下换手率，即上一期和这一期的股票重合度
            # ==============================
            # # 第一期没有换手率
            # if len(df_last) == 0:
            #     df_exchange = pd.DataFrame(data=[0]*self.n_group,
            #                                index=list(np.arange(1, self.n_group+1)),
            #                                columns=[adj_day])
            # else:
            #     ex_list = []
            #     for i in np.arange(1, self.n_group+1):
            #         stocks_now = df_2[df_2['group'] == i].index
            #         stocks_last = df_last[df_last['group'] == i].index
            #         print(stock_last)
            #         retain_rate = len(set(stocks_last).intersection(set(stocks_now))) / len(stocks_last)
            #         exchange_rate = 1 - retain_rate
            #         ex_list.append(exchange_rate)
            #     df_exchange = pd.DataFrame(data=ex_list,
            #                                index=list(np.arange(1, self.n_group+1)),
            #                                columns=[adj_day])
            # ex_results = pd.concat([ex_results, df_exchange], axis=1)
            # df_last = df_2.copy()

        df_ic = pd.DataFrame(data=[ic_list, rank_ic_list],
                                   index=['ic', 'rank_ic'],
                                   columns=adj_date).T

        df_group_ic = pd.DataFrame(data=group_ic_list,
                                   columns=['group_ic'],
                                   index=adj_date)

        results = {
            'adj_period': adj_periods,
            'ic': df_ic,
            'group_ic': df_group_ic,
            'group_inner_ic': df_group_inner_ic,
            'ret': ret_results,
            # 'excess_ret': excess_ret_results,
            'group_num': df_group_num,
            'exchange': ex_results
        }

        return results

    def calc_multiple_periods(self):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = [executor.submit(self.calc_results, i)
                       for i in self.list_periods]
        results = [f.result() for f in concurrent.futures.as_completed(results)]
        results = sorted(results, key = lambda e: e.__getitem__('adj_period'))  # 按照adj_period排序
        self.results = results
        return results

    # ==============================
    # ==============================
    # IC的分析
    # ==============================
    # ==============================
    # 计算IC和分组IC的样本统计量
    def summary_ic_table(self, start=False, end=False, kind='rank_ic'):
        """
        对于ic的综合描述，包括均值，ir，显著性，偏度，峰度
        :param start: 开始日期，如’2021-01-01‘
        :param end: 结束日期，如’2022-01-01‘
        :param kind: 计算ic的方式，默认是spearman
        :return: 两个dataframe
        """
        if not start:
            start = self.factor.index[0]
        if not end:
            end = self.factor.index[-1]
        summary_ic = pd.DataFrame()
        summary_group_ic = pd.DataFrame()
        for result in self.results:
            adj_period = result['adj_period']
            # ==============================
            # IC的分析
            # ==============================
            ic = result['ic'].copy()
            ic = ic[(ic.index >= start) & (ic.index <= end)]
            # 计算ic_mean
            ic_mean = ic.mean()
            ic_std = ic.std(ddof=1)
            # 计算ic_ir
            ic_ir = np.abs(ic_mean) / ic_std
            # 偏度
            ic_skew = ic.skew()
            # 峰度
            ic_kurtosis = ic.kurtosis()
            # 检验是否显著为0
            t_0 = np.sqrt(len(ic)) * ic_mean / ic_std
            p_0 = np.abs(t_0).apply(lambda x: (1-scipy.stats.t.cdf(x, len(ic)-1)) / 2)
            df_tmp1 = pd.DataFrame(data=[ic_mean, ic_ir,
                                         ic_skew, ic_kurtosis,
                                         t_0, p_0],
                                   index=['IC Mean', 'IC IR',
                                          'IC Skew', 'IC Kurtosis',
                                          't: IC!=0', 'p: IC!=0'])
            df_tmp1.columns = [[f'period_{adj_period}', f'period_{adj_period}'],
                               ['IC', 'rank IC']]
            # # 检验是否显著大于0.03
            # t_003 = np.sqrt(len(ic)) * (ic_mean - 0.03) / ic_std
            # p_003 = np.abs(t_003).apply(lambda x: 1-scipy.stats.t.cdf(x, len(ic)-1))
            # # 检验是否显著小于-0.03
            # t_003_ = np.sqrt(len(ic)) * (ic_mean + 0.03) / ic_std
            # p_003_ = np.abs(t_003_).apply(lambda x: 1-scipy.stats.t.cdf(x, len(ic)-1))
            # df_tmp1 = pd.DataFrame(data=[ic_mean, ic_ir, ic_skew, ic_kurtosis,
            #                              t_0, p_0,
            #                              t_003, p_003,
            #                              t_003_, p_003_],
            #                        index=['IC Mean', 'IC IR', 'IC Skew', 'IC Kurtosis',
            #                               't: IC!=0', 'p: IC!=0',
            #                               't: IC>0.03', 'p: IC>0.03',
            #                               't: IC<-0.03', 'p: IC<-0.03'])
            # df_tmp1.columns = [[f'period_{adj_period}', f'period_{adj_period}'],
            #                    ['IC', 'rank IC']]
            df_tmp1 = df_tmp1.round(4)
            summary_ic = pd.concat([summary_ic, df_tmp1], axis=1)

            # ==============================
            # Group_IC的分析
            # ==============================
            group_ic = result['group_ic'].copy()
            group_ic = group_ic[(group_ic.index >= start) & (group_ic.index <= end)]
            # 计算group_ic_mean
            group_ic_mean = group_ic.mean()
            group_ic_std = group_ic.std(ddof=1)
            # 计算group_ic_ir
            group_ic_ir = group_ic_mean / group_ic_std
            # 偏度
            group_ic_skew = group_ic.skew()
            # 峰度
            group_ic_kurtosis = group_ic.kurtosis()
            # 检验是否显著为0
            t_0 = group_ic_ir * np.sqrt(len(group_ic))
            p_0 = np.abs(t_0).apply(lambda x: (1-scipy.stats.t.cdf(x, len(ic)-1)) / 2)
            df_tmp2 = pd.DataFrame(data=[group_ic_mean, group_ic_ir,
                                         group_ic_skew, group_ic_kurtosis,
                                         t_0, p_0],
                                   index=['IC Mean', 'IC IR', 'IC Skew', 'IC Kurtosis',
                                          't: IC!=0', 'p: IC!=0'])
            df_tmp2.columns = [f'period_{adj_period}: group rank IC']
            df_tmp2 = df_tmp2.round(4)
            summary_group_ic = pd.concat([summary_group_ic, df_tmp2], axis=1)

        dict1 = {}
        if kind == 'rank_ic':
            df_ic = summary_ic.iloc[:, list(np.arange(0, 2*len(self.results), 2)+1)]
            df_ic.columns = [f'{x[0]}: {x[1]}' for x in df_ic.columns]
            dict1['ic'] = df_ic
            dict1['group_ic'] = summary_group_ic
            return dict1
        elif kind == 'ic':
            df_ic = summary_ic.iloc[:, list(np.arange(0, 2*len(self.results), 2))]
            df_ic.columns = [f'{x[0]}: {x[1]}' for x in df_ic.columns]
            dict1['ic'] = df_ic
            dict1['group_ic'] = summary_group_ic
            return dict1

    # 画ic的变化柱状图，可以按年或者按月
    def plot_ic_bar(self, frequency='Y', kind='rank_ic'):
        df_ic = pd.DataFrame()
        for result in self.results:
            ic = result['ic'].loc[:, kind].copy()
            ic = ic.resample(frequency, axis=0, label='right',
                             closed='right', kind='period').mean().to_frame()
            ic = (100 * ic).round(2)
            ic.columns = [result['adj_period']]
            df_ic = pd.concat([df_ic, ic], axis=1)
        df_ic.plot(kind='bar', figsize=(10, 5))
        plt.title('IC (%)', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()

    # 画多空收益率按30天滚动和每日的IC
    # 在legend中，plot的优先级高于bar
    def plot_daily_ic(self, kind='rank_ic'):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ic = self.results[i]['ic'][kind].to_frame()
            ic = (100 * ic).round(2)
            ic['30d'] = ic[kind].rolling(window=int(20//adj_period)+1).mean()
            ax = fig.add_subplot(100*(len(self.results)) + 10 + (i+1))
            ax.bar(ic.index, ic[kind], alpha=0.6, width=2)
            ax.plot(ic.index, ic['30d'], alpha=1, color='red')
            ax.set_title(f'Adj_periods_{adj_period}: {kind} (%)', fontsize=15)
            ax.legend(['30d', 'daily'], loc='upper left')  # 先写plot的，再写bar的
            ax.fill_between(ic.index, 3, -3,color='yellow',alpha=0.4)
            ax.set_ylim(-40, 40)
        plt.tight_layout()
        plt.show()

    # plot_monthly_IC
    def plot_monthly_ic(self, kind='rank_ic'):
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ic = self.results[i]['ic'][kind].to_frame()
            ic = ic.resample('M', axis=0, label='right',
                             closed='right', kind='period').mean()
            ic.columns = [kind]
            ic['year'] = ic.index.year
            ic['month'] = ic.index.month
            ic = ic.pivot_table(index=['year'], columns=['month'],
                                values=[kind])
            ic.columns = [x[1] for x in ic.columns]
            ic = (100 * ic).round(2)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.imshow(ic.values, cmap="YlGn")
            # Rotate the tick labels and set their alignment.
            ax.set_xticks(np.arange(len(ic.columns)), labels=ic.columns, fontsize=15)
            ax.set_yticks(np.arange(len(ic.index)), labels=ic.index, fontsize=15)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            #          rotation_mode="anchor")
            for j in range(len(ic.index)):
                for k in range(len(ic.columns)):
                    text = ax.text(k, j,ic.values[j, k],
                                        ha="center", va="center", color="black")
            ax.set_title(f'Adj_period_{adj_period}: Monthly IC (%)')
        plt.tight_layout()
        plt.show()
        plt.style.use('_mpl-gallery')

    # ==============================
    # 收益率分析
    # ==============================
    # plot_monthly
    def plot_monthly_ret(self):
        plt.style.use('default')
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'].T.copy()
            ret = (ret.iloc[:, -1] - ret.iloc[:, 0]).to_frame()  # 用普通收益率算多空
            ret = np.log(1+ret)  # 转化为对数收益率方便加和
            ret = ret.resample('M', axis=0, label='right',
                               closed='right', kind='period').sum()
            ret = np.exp(ret) - 1  # 计算出了每个月的普通收益率
            ret.columns = ['ret']
            ret['year'] = ret.index.year
            ret['month'] = ret.index.month
            ret = ret.pivot_table(index=['year'], columns=['month'],
                                  values=['ret'])
            ret.columns = [x[1] for x in ret.columns]
            ret = (100 * ret).round(2)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.imshow(ret.values, cmap="YlGn")
            # Rotate the tick labels and set their alignment.
            ax.set_xticks(np.arange(len(ret.columns)), labels=ret.columns, fontsize=15)
            ax.set_yticks(np.arange(len(ret.index)), labels=ret.index, fontsize=15)
            # plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            #          rotation_mode="anchor")

            for j in range(len(ret.index)):
                for k in range(len(ret.columns)):
                    text = ax.text(k, j,ret.values[j, k],
                                        ha="center", va="center", color="black")
            ax.set_title(f'Adj_period_{adj_period}: Long Short Monthly Return (%)')
        plt.tight_layout()
        plt.show()
        plt.style.use('_mpl-gallery')

    # 画每一组的收益率
    def plot_layer_ret_bar(self, excess=True):
        df_ret = pd.DataFrame()
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'].T.copy()
            ret.columns = list(range(1, self.n_group+1))
            ret.index = pd.to_datetime(ret.index)
            if excess:
                ret = ret - ret.mean(axis=1).values.reshape(len(ret), 1)
            ret = np.cumprod(1+ret, axis=0) - 1
            ret = (1 + ret) ** (1 / len(ret)) - 1
            # ret.dropna(axis=0, inplace=True)
            ret = 10000 * ret.iloc[-1, :].to_frame()
            ret.columns = [adj_period]
            df_ret = pd.concat([df_ret, ret], axis=1)
        df_ret.plot(kind='bar', figsize=(10, 5))
        plt.title('Group Return', fontsize=15)
        plt.ylabel('daily return (bps)', fontsize=15)
        plt.legend(fontsize=15)
        plt.show()

    # 画分层收益率的图
    def plot_layer_ret(self, excess=True):
        cmap = cm.get_cmap("RdYlGn")
        cmap = cmap(np.linspace(0, 1, self.n_group))
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            ret = self.results[i]['ret'].T.copy()
            if excess:
                ret = ret - ret.mean(axis=1).values.reshape(len(ret), 1)
            ret = np.cumprod(1+ret, axis=0)
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            for j in range(len(ret.columns)):
                ax.plot(ret.index, ret.iloc[:, j].values, color=cmap[j], alpha=1)
            ax.legend(ret.columns, loc='upper left', fontsize=10)
            adj_period = self.results[i]['adj_period']
            ax.set_title(f'Adj_periods_{adj_period}: Net Value for Groups', fontsize=15)
            ax.axhline(1, linestyle='--', c='grey')
        plt.tight_layout()
        plt.show()

    # 画多空收益组合的图
    def plot_long_short_ret(self):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            ret = self.results[i]['ret'].T.copy()
            ret = np.cumprod(1 + (ret.iloc[:, -1] - ret.iloc[:, 0]), axis=0).to_frame()
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.plot(ret.index, ret.iloc[:, 0], alpha=1)
            ax.legend(['Net_Value'], loc='upper left', fontsize=10)
            adj_period = self.results[i]['adj_period']
            ax.set_title(f'Adj_periods_{adj_period}: Long Short Portfolio Net Value', fontsize=15)
            ax.axhline(1, linestyle='--', c='grey')
        plt.tight_layout()
        plt.show()

    # 画多空收益率按30天滚动和每日的收益率
    def plot_daily_ret(self):
        fig = plt.figure(figsize=(10, 5*len(self.results)))
        for i in range(len(self.results)):
            adj_period = self.results[i]['adj_period']
            ret = self.results[i]['ret'].T.copy()
            ret = (ret.iloc[:, -1] - ret.iloc[:, 0]).to_frame()
            ret.columns = ['daily']
            ret = (100 * ret).round(2)
            ret['30d'] = ret['daily'].rolling(window=(1 + 20//adj_period)).mean()
            ax = fig.add_subplot(100*len(self.results) + 10 + (i+1))
            ax.bar(ret.index, ret['daily'], alpha=0.6, width=2)
            ax.plot(ret.index, ret['30d'], alpha=1, color='red')
            ax.legend(['30d', 'daily'], loc='upper left')  # 先写plot的，再写bar的
            adj_period = self.results[i]['adj_period']
            ax.set_title(f'Adj_periods_{adj_period}: Long Short Portfolio Returns', fontsize=15)
            # axes[i].axhline(0, linestyle='--', c='grey')
            ax.set_ylabel('return (%)')
        plt.tight_layout()
        plt.show()

    # ==============================
    # ==============================
    # 快速分析
    # ==============================
    # ==============================
    def fast_analysis(self):
        self.calc_multiple_periods()
        # ic的部分
        print('############### IC ###############')
        summary_ic_table = self.summary_ic_table()
        print('--------------- IC performance ---------------: ')
        print(summary_ic_table['ic'])
        # print('\nGroup IC performance: ')
        # print(summary_ic_table['group_ic'])
        print('\n--------------- IC yearly ---------------')
        self.plot_ic_bar('Y', kind='rank_ic')
        print('\n--------------- IC monthly ---------------')
        self.plot_monthly_ic(kind='rank_ic')
        print('\n--------------- IC daily ---------------')
        self.plot_daily_ic(kind='rank_ic')

        # return的部分
        print('\n############### Return ###############')
        print('\n--------------- Long Short portfolio ---------------')
        self.plot_long_short_ret()
        print('\n--------------- Long Short Monthly Return ---------------')
        self.plot_monthly_ret()
        print('\n--------------- Returns in each layer ---------------')
        self.plot_layer_ret_bar(excess=False)
        self.plot_layer_ret(excess=False)
        print('\n--------------- Excess Returns in each layer ---------------')
        self.plot_layer_ret_bar(excess=True)
        self.plot_layer_ret(excess=True)
        print('\n--------------- Return daily ---------------')
        self.plot_daily_ret()





