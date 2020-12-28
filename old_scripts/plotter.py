def plot_results(df,xaxis,yaxis,lines='alg',filters=None,**plotattrs):
    '''
    Averages accross all columns not specified in constants
    lines - string, list, or dataframe to select which lines to plot
        string or list chooses column(s) from dataframe and plots all unique
            entries/combinations as a separate line.
        dataframe plots each row from dataframe as a separate line
    filters - dict where key is attribute and value is list of permissible
    
    '''
    #Set Default Plot Attributes
    CI_style = plotattrs.setdefault('CI_style','polygon') #Confidence interval style 'polygon' or 'errorbar'
    colormap = plotattrs.setdefault('colormap','tab10')
    linestyles = plotattrs.setdefault('linestyles',['-'])
    xscale = plotattrs.setdefault('xscale','linear') #'log' or 'linear'
    yscale = plotattrs.setdefault('yscale','linear') #'log' or 'linear'
    linewidth = plotattrs.setdefault('linewidth',3)
    incl_exp = plotattrs.setdefault('incl_exp',False)
    incl_rand = plotattrs.setdefault('incl_rand',False)
    legend_on = plotattrs.setdefault('legend_on',True)
    ylabel_on = plotattrs.setdefault('ylabel_on',True)
    save_dir = plotattrs.setdefault('save_dir',None)
    env_id = plotattrs.setdefault('env_id',df['env_id'][0])
    exp_name = plotattrs.setdefault('exp_name','')
    legend = plotattrs.setdefault('legend',None)

    #Make sure filter values are lists, even if they only have one item
    filters = {k:([filters[k]] if type(filters[k]) is not list else filters[k]) for k in filters} if filters is not None else None

    # Gather lines and filter for desired attrs
    lines = [lines] if type(lines) is str else lines
    lines_df = df[lines].drop_duplicates().dropna() if type(lines) is list else lines
    if not incl_exp:
        lines_df = lines_df[~(lines_df['alg'].isin(['Expert']))]
    if not incl_rand:
        lines_df = lines_df[~(lines_df['alg'].isin(['Random']))]
    lines_df = lines_df.sort_values('alg')
    if filters is not None:
        df = df[(df[filters.keys()].isin(filters)).all(axis=1) | (df['alg'].isin(['Expert']) & incl_exp) | (df['alg'].isin(['Random']) & incl_rand)]

    #Set colormap, linestyles,axis labels, legend label formats, title
    if colormap in DISCRETE_COLORMAPS:
        cs = plt.cm.get_cmap(colormap).colors #Discrete colormaps
    else: #Contiuous colormaps
        cs = [plt.cm.get_cmap(colormap)(val) for val in np.linspace(1,0,len(lines_df))] 
    lss = [ls for i in range(len(lines_df)) for ls in linestyles]
    xlabel = {'N_E_traj':'Number of Expert Trajectories',
              'N_E_samp':'Number of Expert Samples'}.get(xaxis,xaxis)
    ylabel = {'reward':'On-Policy Reward',
              'loss_train':'Training Loss',
              'loss_test':'Validation Loss',
              'w_ESS':'Effective Sample Size'}.get(yaxis,yaxis)
    leg_format = lambda col: {'total_opt_steps':'{:.2g}'}.get(col,'{}')
    leg_label = lambda col: {'total_opt_steps':'# opt','entropy_coeff':'w_H',
                             'model_reg_coeff':'reg','learning_rate':'lr'}.get(col,col)

    title = plotattrs.setdefault('title',' '.join([env_id,exp_name,ylabel]))

    #Add lines
    figname = '-'.join([env_id,exp_name,xaxis,yaxis])
    plt.figure(figname)
    lines_df = pd.DataFrame(lines_df).reset_index(drop=True) #Handle single line
    for i,line in lines_df.iterrows():
        label = ', '.join([leg_label(k)+' = '+leg_format(k).format(v) if k!='alg' else v for k,v in line.items()])
        label = line['alg'] if line['alg'] in ['Expert','Random'] else label
        label = legend[i] if legend is not None else label
        df_line = df[(df[line.keys()]==line.values).all(axis=1)].sort_values(xaxis)
        x = df_line[xaxis].drop_duplicates().to_numpy()
        n = np.array([len(df_line[df_line[xaxis]==xi][yaxis]) for xi in x])
        y = np.array([df_line[df_line[xaxis]==xi][yaxis].mean() for xi in x])
        yerr = np.array([df_line[df_line[xaxis]==xi][yaxis].std() for xi in x])
        if (yaxis=='reward') and (n==1).all() and ('reward_std' in df_line.columns):
            yerr = np.squeeze(np.array([df_line[df_line[xaxis]==xi]['reward_std'].to_numpy() for xi in x]))

        if CI_style == 'errorbar':
            plt.errorbar(x,y,yerr=yerr,c=cs[i],ls=lss[i],label=label,lw=linewidth)
        if CI_style == 'polygon':
            plt.plot(x,y,c=cs[i],ls=lss[i],label=label,lw=linewidth)
            xy = np.hstack((np.vstack((x,y + yerr)),np.fliplr(np.vstack((x,y - yerr))))).T
            plt.gca().add_patch(Polygon(xy=xy, closed=True, fc=cs[i], alpha=.2))
        

    plt.xscale(xscale)
    plt.yscale(yscale)
    for axis in [plt.gca().xaxis, plt.gca().yaxis]:
        axis.set_major_formatter(ScalarFormatter())
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if legend_on:
        plt.legend()
    if save_dir is not None:
        plt.savefig(os.path.join(save_dir,figname))