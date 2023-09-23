# plot function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import pandas as pd
import numpy as np

from .portfolio_analysis import ptf_analysis as ptfa

def plot(portfolio:ptfa, pattern:str ='scatter', alpha:bool =False, percentage:bool =True, figsize:tuple =(14,7), pos: float=0.0, **kwargs):
    if type(portfolio).__name__ == 'Univariate':
        plot_type = 'uni'
    elif type(portfolio).__name__ == 'Bivariate':
        plot_type = 'bi'
    
    if alpha == False:
        average_result, ttest = portfolio.return_average_ttest(alpha=alpha)
        alpha_mat = np.zeros(np.shape(average_result))
        alpha_tvalue = np.zeros(np.shape(average_result))
    elif alpha == True: 
        average_result, ttest, alpha_mat, alpha_tvalue = portfolio.return_average_ttest(alpha=alpha)
    
    if percentage == True:
        average_result = 100 * average_result
        alpha_mat = 100 * alpha_mat
    
    if plot_type == 'uni':
        row = len(average_result)
        if len(average_result) >= 2:
            col_names = ['Low'] + [str(i+1) for i in range(len(average_result)-3)] + ['High']
        
        if alpha == True:
            alpha_mat = alpha_mat[:, 0]
            alpha_tvalue = alpha_tvalue[:, 0]
        
        data = {'portfolio': col_names, 'ave_re': average_result[:-1], 'ttest': ttest[0][:-1], 'alpha':alpha_mat[:-1], 'alpha_tvalue':alpha_tvalue[:-1]}
        data = pd.DataFrame(data)

        if pattern == 'scatter':
            if alpha == False:
                fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.scatterplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
            elif alpha == True:
                fig, ax = plt.subplots(4, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.scatterplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
                sns.scatterplot(data=data, x='portfolio', y='alpha', ax=ax[2], **kwargs)
                sns.scatterplot(data=data, x='portfolio', y='alpha_tvalue', ax=ax[3], **kwargs)
        elif pattern == 'bar':
            if alpha == False:
                fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=500)
                sns.barplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.barplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
            elif alpha == True:
                fig, ax = plt.subplots(4, 1, figsize=figsize, dpi=500)
                sns.barplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.barplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
                sns.barplot(data=data, x='portfolio', y='alpha', ax=ax[2], **kwargs)
                sns.barplot(data=data, x='portfolio', y='alpha_tvalue', ax=ax[3], **kwargs)
        elif pattern == 'line':
            if alpha == False:
                fig, ax = plt.subplots(2, 1, figsize=figsize, dpi=500)
                sns.lineplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.lineplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
            elif alpha == True:
                fig, ax = plt.subplots(4, 1, figsize=figsize, dpi=500)
                sns.lineplot(data=data, x='portfolio', y='ave_re', ax=ax[0], **kwargs)
                sns.lineplot(data=data, x='portfolio', y='ttest', ax=ax[1], **kwargs)
                sns.lineplot(data=data, x='portfolio', y='alpha', ax=ax[2], **kwargs)
                sns.lineplot(data=data, x='portfolio', y='alpha_tvalue', ax=ax[3], **kwargs)
        else: raise IOError
    
    elif plot_type == 'bi':
        row, col = np.shape(average_result)
        data = pd.DataFrame(columns=['row', 'col','portfolio', 'ave_re', 'ttest', 'alpha', 'alpha_tvalue'])
        for i in range(row-1):
            for j in range(col-1):
                temp_data = pd.DataFrame({'row':i, 'col':j, 'portfolio': 'Row'+str(i+1)+'Col'+str(j+1), 'ave_re': average_result[i, j], 'ttest': ttest[0][i, j], 'alpha': alpha_mat[i, j], 'alpha_tvalue': alpha_tvalue[i, j]}, index=['Row'+str(i+1)+'Col'+str(j+1)])
                data = pd.concat([data, temp_data])

        if pattern == '3d':
            if alpha == False:
                fig = plt.figure('ave_re', figsize=figsize, dpi=500)
                ax1 = fig.add_subplot(121, projection='3d')
                ax2 = fig.add_subplot(122, projection='3d')
                
                # average_result
                for i in range(len(data['row'].index)):
                    color = (data['ave_re'][i] - min(data['ave_re']))/(max(data['ave_re'])-min(data['ave_re']))
                    ax1.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['ave_re'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax1.set_xlabel('Row')
                ax1.set_ylabel('Col')
                ax1.set_zlabel('ave_re')
                
                ax1.set_xticks([i for i in range(row)])
                ax1.set_yticks([i for i in range(col)])
                
                if min(data['ave_re']) < 0:
                    zmin = 1.2 * min(data['ave_re'])
                else:
                    zmin = 0.8 * min(data['ave_re'])
                
                if max(data['ave_re']) < 0:
                    zmax = 0.8 * max(data['ave_re'])
                else:
                    zmax = 1.2 * max(data['ave_re'])

                ax1.set_zlim(zmin, zmax)

                # ttest
                for i in range(len(data['row'].index)):
                    color = (data['ttest'][i] - min(data['ttest']))/(max(data['ttest'])-min(data['ttest']))
                    ax2.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['ttest'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax2.set_xlabel('Row')
                ax2.set_ylabel('Col')
                ax2.set_zlabel('ttest')

                ax2.set_xticks([i for i in range(row)])
                ax2.set_yticks([i for i in range(col)])

                if min(data['ttest']) < 0:
                    zmin = 1.2 * min(data['ttest'])
                else:
                    zmin = 0.8 * min(data['ttest'])
                
                if max(data['ttest']) < 0:
                    zmax = 0.8 * max(data['ttest'])
                else:
                    zmax = 1.2 * max(data['ttest'])

                ax2.set_zlim(zmin, zmax)

            elif alpha == True:
                fig = plt.figure('ave_re', figsize=figsize, dpi=500)
                ax1 = fig.add_subplot(141, projection='3d')
                ax2 = fig.add_subplot(142, projection='3d')
                ax3 = fig.add_subplot(143, projection='3d')
                ax4 = fig.add_subplot(144, projection='3d')

                # average result
                for i in range(len(data['row'].index)):
                    color = (data['ave_re'][i] - min(data['ave_re']))/(max(data['ave_re'])-min(data['ave_re']))
                    ax1.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['ave_re'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax1.set_xlabel('Row')
                ax1.set_ylabel('Col')
                ax1.set_zlabel('ave_re')

                ax1.set_xticks([i for i in range(row)])
                ax1.set_yticks([i for i in range(col)])

                if min(data['ave_re']) < 0:
                    zmin = 1.2 * min(data['ave_re'])
                else:
                    zmin = 0.8 * min(data['ave_re'])
                
                if max(data['ave_re']) < 0:
                    zmax = 0.8 * max(data['ave_re'])
                else:
                    zmax = 1.2 * max(data['ave_re'])

                ax1.set_zlim(zmin, zmax)

                # ttest
                for i in range(len(data['row'].index)):
                    color = (data['ttest'][i] - min(data['ttest']))/(max(data['ttest'])-min(data['ttest']))
                    ax2.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['ttest'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax2.set_xlabel('Row')
                ax2.set_ylabel('Col')
                ax2.set_zlabel('ttest')

                ax2.set_xticks([i for i in range(row)])
                ax2.set_yticks([i for i in range(col)])

                if min(data['ttest']) < 0:
                    zmin = 1.2 * min(data['ttest'])
                else:
                    zmin = 0.8 * min(data['ttest'])
                
                if max(data['ttest']) < 0:
                    zmax = 0.8 * max(data['ttest'])
                else:
                    zmax = 1.2 * max(data['ttest'])

                ax2.set_zlim(zmin, zmax)
                
                # alpha
                for i in range(len(data['row'].index)):
                    color = (data['alpha'][i] - min(data['alpha']))/(max(data['alpha'])-min(data['alpha']))
                    ax3.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['alpha'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax3.set_xlabel('Row')
                ax3.set_ylabel('Col')
                ax3.set_zlabel('alpha')
                
                ax3.set_xticks([i for i in range(row)])
                ax3.set_yticks([i for i in range(col)])

                if min(data['alpha']) < 0:
                    zmin = 1.2 * min(data['alpha'])
                else:
                    zmin = 0.8 * min(data['alpha'])
                
                if max(data['alpha']) < 0:
                    zmax = 0.8 * max(data['alpha'])
                else:
                    zmax = 1.2 * max(data['alpha'])

                ax3.set_zlim(zmin, zmax)

                # alpha tvalue
                for i in range(len(data['row'].index)):
                    color = (data['alpha_tvalue'][i] - min(data['alpha_tvalue']))/(max(data['alpha_tvalue'])-min(data['alpha_tvalue']))
                    ax4.bar3d(data['row'][i], data['col'][i], pos, 1, 1, data['alpha_tvalue'][i], color=np.array([1, 0.9, 0.9*color]), **kwargs)
                #ax.bar3d(data['row'], data['col'], 0, 1, 1, data['ave_re'], shade=True, cmap='rainbow')
                ax4.set_xlabel('Row')
                ax4.set_ylabel('Col')
                ax4.set_zlabel('alpha_tvalue')
                
                ax4.set_xticks([i for i in range(row)])
                ax4.set_yticks([i for i in range(col)])

                if min(data['alpha_tvalue']) < 0:
                    zmin = 1.2 * min(data['alpha_tvalue'])
                else:
                    zmin = 0.8 * min(data['alpha_tvalue'])
                
                if max(data['alpha_tvalue']) < 0:
                    zmax = 0.8 * max(data['alpha_tvalue'])
                else:
                    zmax = 1.2 * max(data['alpha_tvalue'])

                ax4.set_zlim(zmin, zmax)

                plt.show()
        
        elif pattern == 'scatter':
            if alpha == False:
                fig1, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='ave_re', style='col', hue='col', ax=ax1, legend='full', **kwargs)
                ax1.set_xticks([i for i in range(row-1)])
                
                fig2, ax2 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='ttest', style='col', hue='col', ax=ax2, legend='full', **kwargs)
                ax2.set_xticks([i for i in range(row-1)])
                
                fig3, ax3 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='ave_re', style='row', hue='row', ax=ax3, legend='full', **kwargs)
                ax3.set_xticks([i for i in range(col-1)])

                fig4, ax4 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='ttest', style='row', hue='row', ax=ax4, legend='full', **kwargs)
                ax4.set_xticks([i for i in range(col-1)])

            elif alpha == True:
                fig1, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='ave_re', style='col', hue='col', ax=ax1, legend='full', **kwargs)
                ax1.set_xticks([i for i in range(row-1)])

                fig2, ax2 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='ttest', style='col', hue='col', ax=ax2, legend='full', **kwargs)
                ax2.set_xticks([i for i in range(row-1)])
                
                fig3, ax3 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='ave_re', style='row', hue='row', ax=ax3, legend='full', **kwargs)
                ax3.set_xticks([i for i in range(col-1)])
                
                fig4, ax4 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='ttest', style='row', hue='row', ax=ax4, legend='full', **kwargs)
                ax4.set_xticks([i for i in range(col-1)])

                fig5, ax5 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='alpha', style='col', hue='col', ax=ax5, legend='full', **kwargs)
                ax5.set_xticks([i for i in range(row-1)])
                
                fig6, ax6 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='row', y='alpha_tvalue', style='col', hue='col', ax=ax6, legend='full', **kwargs)
                ax6.set_xticks([i for i in range(row-1)])
                
                fig7, ax7 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='alpha', style='row', hue='row', ax=ax7, legend='full', **kwargs)
                ax7.set_xticks([i for i in range(col-1)])

                fig8, ax8 = plt.subplots(1, 1, figsize=figsize, dpi=500)
                sns.scatterplot(data=data, x='col', y='alpha_tvalue', style='row', hue='row', ax=ax8, legend='full', **kwargs)
                ax8.set_xticks([i for i in range(col-1)])











    





