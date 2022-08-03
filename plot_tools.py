import matplotlib
matplotlib.rcParams.update({'font.size': 16,
                            #'font.family':'sans-serif',
                            #'font.sans-serif':['Arial'],
                            'font.family': 'STIXGeneral',
                            'mathtext.fontset': 'stix',
                            'axes.titleweight':'bold',
                            'text.usetex': True,
                            'text.latex.preamble':r"""\usepackage{amsmath}
                                                  \usepackage{amssymb}"""})


colors = {
    'dark grey':[0.5]*3+[1.0]#'#FF9d9d9d'
}
style_colors = {
    'spines': colors['dark grey'],
    'ticks': colors['dark grey']
}
def ax_style(ax,left_spine_lim=None,keep=['left','bottom']):
    """
    Removes spines except those in 'keep'.
    Colors them dark grey.
    
    """
    all = ['top','right','left','bottom']

    for side in list(set(all)-set(keep)):
        ax.spines[side].set_visible(False)#.set_color('none'
    for side in keep:
        ax.spines[side].set_color(style_colors['spines'])
        ax.spines[side].set_linewidth(1.3)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which=u'both', color=style_colors['spines'])
    if left_spine_lim is not None:
        ax.spines['left'].set_bounds(*left_spine_lim)
        

def ax_style(ax,spine_lims=[None]*4,keep=['left','bottom']):
    """
    Removes spines except those in 'keep'.
    Colors them dark grey.
    """
    all_sides = ['top','right','left','bottom']

    for side in list(set(all_sides)-set(keep)):
        ax.spines[side].set_visible(False)#.set_color('none'
    for side in keep:
        ax.spines[side].set_color(style_colors['spines'])
        ax.spines[side].set_linewidth(1.3)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        ax.tick_params(axis='both', which=u'both', color=style_colors['spines'])
    for side,spine_lim in zip(all_sides,spine_lims):
        if spine_lim is not None:
            ax.spines[side].set_bounds(*spine_lim)


def fig_labels(fig,axes):
    alphabets = list(string.ascii_uppercase)
    for ax,alpbt in zip(axes,alphabets):
        ax.set_title(alpbt,va='bottom',ha='right',loc='left',fontweight='bold')
        
def subplot_org(fig,
                axs,
                xlims=None,
                ylims=None,
                xlabel=None,
                ylabel=None,
                xticks=None,
                yticks=None,
                xticklabels=None,
                yticklabels=None,
                use_ticks_as_ticklabels=[True,True]):
    if use_ticks_as_ticklabels[0]:
        xticklabels = xticks
    if use_ticks_as_ticklabels[1]:
        yticklabels = yticks
    print(xticklabels)
    for i in range(axs.shape[0]):
        for j in range(axs.shape[1]):
            ax = axs[i,j]
            ax_style(ax)
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)

            if xticks is not None:
                ax.set_xticks(xticks)
            if yticks is not None:
                ax.set_yticks(yticks)
            if i == axs.shape[0]-1:
                if  j != 0:
                    ax.set_yticklabels([])
                ax.set_xlabel(xlabel)                   
                ax.set_xticklabels(xticklabels)
                
            if j == 0:
                if i != axs.shape[0]-1:
                    ax.set_xticklabels([])
                ax.set_ylabel(ylabel)
                ax.set_yticklabels(yticklabels)
            if not (i == axs.shape[0]-1) and not (j==0):
                #ax.set_title(xticklabels)
                ax.set_xticklabels([])
                ax.set_yticklabels([])