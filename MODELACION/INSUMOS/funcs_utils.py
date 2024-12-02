from IPython.display import HTML, display
from ipywidgets import Button, Layout
from matplotlib.ticker import PercentFormatter
from scipy import stats
from statsmodels.graphics.mosaicplot import mosaic
from statsmodels.stats import diagnostic
from scipy.spatial.distance import cdist
from scipy.spatial import distance_matrix

import scipy.stats as sta
import unicodedata
import ipywidgets as widgets
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def print_spread_measures(col):

    notnull_col = col[~np.isnan(col)]
    percentiles = dict(notnull_col.quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    measures = {}
    measures["Desv. Est."] = notnull_col.std()
    measures['Rango'] = notnull_col.max() - notnull_col.min()
    measures['Rango IQ'] = stats.iqr(notnull_col)
    measures['Dif. Abs. Media'] = np.mean(np.abs(notnull_col - notnull_col.mean()))
    measures['Dif. Abs. Mediana'] = np.median(np.abs(notnull_col - notnull_col.median()))
    measures['Coef. Var.'] = stats.variation(notnull_col)
    measures['QCD'] = (percentiles[0.75] - percentiles[0.25]) / (percentiles[0.75] + percentiles[0.25])
    statistics = pd.DataFrame.from_dict(data=measures, orient='index', columns=['Resultado'])
    statistics.index.names = ['Medida']
    display(HTML("<h3>Dispersión</h3><br>"))
    display(widgets.HTML(statistics.style\
                        .format({'Resultado':'{:,.2f}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() ))

  
def print_one_way_counts_table(col):
    col= pd.Series(np.where(col.isnull(), 'Sin dato', 'Con dato'))
    #freq_table = one_way_table(col=col_count_data, idx_name=idx_name)
    if col is None:
        raise TypeError("Verifique que haya asignado algún objeto al parámetro: 'col'")
    elif col.dtype != 'object':
        raise TypeError("Verifique que el valor asignado al parámetros: 'col', es de tipo: 'object'")
    abs_freq = pd.DataFrame(col.value_counts())
    rel_freq = pd.DataFrame(col.value_counts(normalize=True))
    cum_rel_freq = rel_freq.cumsum()
    
    freq_table = pd.concat([abs_freq, rel_freq, cum_rel_freq], axis=1)
    freq_table.columns = ['Frec.Abs.','Frec.Rel.', 'Frec.Rel.Acum.']
    #freq_table.index.name = idx_name
    freq_table.sort_values(by=['Frec.Abs.'], ascending=False)
    freq_table=  widgets.HTML(freq_table.style\
                        .format({'Frec.Abs.': '{:,}',
                                 'Frec.Rel.': '{:.2%}',
                                 'Frec.Rel.Acum.': '{:.2%}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() )		
    display(HTML("<h3>Conteo de registros</h3><br>"))
    display(freq_table)


def location_measures(col):
    notnull_col = col[~np.isnan(col)]
    percentiles = dict(notnull_col.quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    measures = {}
    measures['Mínimo'] = notnull_col.min()
    measures.update({'Percentil ' + str(int(key * 100)):value for (key, value) in percentiles.items()})
    measures['Máximo'] = notnull_col.max()
    statistics = pd.DataFrame.from_dict(data=measures, orient='index', columns=['Resultado'])
    statistics.index.names = ['Medida']
    return widgets.HTML(statistics.style\
                        .format({'Resultado':'{:,.2f}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() )



def print_location_measures(col):
    notnull_col = col[~np.isnan(col)]
    percentiles = dict(notnull_col.quantile([.01,.05,.1,.25,.5,.75,.9,.95,.99]))
    measures = {}
    measures['Mínimo'] = notnull_col.min()
    measures.update({'Percentil ' + str(int(key * 100)):value for (key, value) in percentiles.items()})
    measures['Máximo'] = notnull_col.max()
    statistics = pd.DataFrame.from_dict(data=measures, orient='index', columns=['Resultado'])
    statistics.index.names = ['Medida']
    display(HTML("<h3>Posición</h3><br>"))
    display(widgets.HTML(statistics.style\
                        .format({'Resultado':'{:,.2f}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() ))

def print_central_tendency_measures(col):
    warnings.filterwarnings("ignore")
    notnull_col = col[~np.isnan(col)]
    percentiles = dict(notnull_col.quantile([.05,.25,.5,.75,.95]))
    measures = {}
    measures['Moda'] = notnull_col.mode()[0]
    measures['Media'] = notnull_col.mean()
    try:
        measures['Media Armónica'] = stats.hmean(notnull_col)
    except:
        measures['Media Armónica'] = np.nan
    try:
        measures['Media Geométrica'] = stats.gmean(notnull_col)
    except:
        measures['Media Geométrica'] = np.nan
    measures['Media Cuadrática'] = np.sqrt(np.sum(np.square(notnull_col)) / notnull_col.count())
    measures['Media Trunc.(5%)'] = stats.trim_mean(a=notnull_col, proportiontocut=.05)
    measures['Media IQ'] = stats.trim_mean(a=notnull_col, proportiontocut=.25)
    measures['Media Wins.(5%)'] = np.mean(stats.mstats.winsorize(notnull_col, limits=0.05))
    measures['Trimedia'] = (percentiles[.25] + 2 * percentiles[.5] + percentiles[.75]) / 4
    measures['Mediana'] = np.median(notnull_col)
    measures['Mid Range'] = (notnull_col.min() + notnull_col.max()) / 2 
    measures['Mid Hinge'] = (percentiles[.25] + percentiles[.75]) / 2
    statistics = pd.DataFrame.from_dict(data=measures, orient='index', columns=['Resultado'])
    statistics.index.names = ['Medida']
    display(HTML("<h3>Tendencia central</h3><br>"))
    display(widgets.HTML(statistics.style\
                        .format({'Resultado':'{:,.2f}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() ))

def print_shape_measures(col):
    notnull_col = col[~np.isnan(col)]
    measures = {}
    measures['Asimetría'] = col.skew()
    measures['Exc.Curtosis'] = col.kurtosis()
    statistics = pd.DataFrame.from_dict(data=measures, orient='index', columns=['Resultado'])
    statistics.index.names = ['Medida']
    display(HTML("<h3>Forma</h3><br>"))
    display(widgets.HTML(statistics.style\
                        .format({'Resultado':'{:,.2f}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() ))

def univar_histogram_plot(col):
    notnull_col = col[~np.isnan(col)]
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(4.8, 3.4))
    sns.distplot(notnull_col, hist=True, kde=True)
    plt.axvline(notnull_col.mean(), color='green', linestyle='dashed', linewidth=2, label="Media")
    plt.axvline(notnull_col.median(), color='blue', linestyle='dashed', linewidth=2, label="Mediana")
    plt.title('Histograma y densidad', fontweight='bold', fontsize=13)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_xticklabels(['{:,.2f}'.format(x) for x in ax1.get_xticks()], fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(notnull_col.name, fontweight='bold', fontsize=11)
    plt.ylabel("Densidad", fontweight='bold', fontsize=11)
    leg=plt.legend(loc='upper right', title='Estadísticos')
    plt.setp(leg.get_title(), fontweight='bold', fontsize=10)
    plt.setp(leg.get_texts(), fontsize=9)
    plt.show()

def univar_violin_plot(col):
    notnull_col = col[~np.isnan(col)]
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    sns.violinplot(x=notnull_col, palette='Blues')
    plt.title('Diagrama de Violín', fontweight='bold', fontsize=13)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_xticklabels(['{:,.2f}'.format(x) for x in ax1.get_xticks()], fontsize=10)
    plt.xlabel(notnull_col.name, fontweight='bold', fontsize=11)
    plt.show()


def univar_box_plot(col):
    notnull_col = col[~np.isnan(col)]
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    sns.boxplot(x=notnull_col, palette='Blues')
    plt.axvline(notnull_col.mean(), color='green', linestyle='dashed', linewidth=2, label='Media')
    plt.title('Diagrama de caja y bigotes', fontweight='bold', fontsize=13)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_xticklabels(['{:,.2f}'.format(x) for x in ax1.get_xticks()], fontsize=10)
    plt.xlabel(notnull_col.name, fontweight='bold', fontsize=11)
    leg=plt.legend(loc='upper right', title='Estadístico')
    plt.setp(leg.get_title(), fontweight='bold', fontsize=10)
    plt.setp(leg.get_texts(), fontsize=9)
    plt.show()


def univar_normal_density(col):
    notnull_col = col[~np.isnan(col)]
    x = np.linspace(notnull_col.min(), notnull_col.max(), 1000)
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(4.8, 3.4))
    sns.distplot(notnull_col, hist=False, kde=True, label='Empírica')
    ax1.plot(x, stats.norm.pdf(x=x, loc=notnull_col.mean(), scale=notnull_col.std()), label='Normal')
    plt.title('Funciones de densidad', fontweight='bold', fontsize=13)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax1.set_xticklabels(['{:,.2f}'.format(x) for x in ax1.get_xticks()], fontsize=10)
    plt.yticks(fontsize=10)
    plt.xlabel(notnull_col.name, fontweight='bold', fontsize=11)
    plt.ylabel("Densidad", fontweight='bold', fontsize=11)
    leg=plt.legend(loc='upper right', title='Distribución')
    plt.setp(leg.get_title(), fontweight='bold', fontsize=10)
    plt.setp(leg.get_texts(), fontsize=9)
    plt.show()


def univar_qq_norm(col):
    notnull_col = col[~np.isnan(col)]
    fig = sm.qqplot(data=notnull_col, fit=True, marker='.', markerfacecolor='w', markeredgecolor='tab:blue', markersize=10, linestyle='--')
    sm.qqline(fig.axes[0], line='45', fmt='k-')
    plt.title('Diagrama cuantil-cuantil', fontweight='bold', fontsize=13)
    plt.yticks(fontsize=10)
    plt.xticks(fontsize=10)
    plt.xlabel('Cuantiles teóricos normales', fontweight='bold', fontsize=11)
    plt.ylabel('Cuantiles muestrales', fontweight='bold', fontsize=11)
    fig.set_size_inches(4.8, 3.4, forward=True)
    plt.show()

def print_normality_tests(col):
    notnull_col = col[~np.isnan(col)]
    try:
        sw_stat, sw_pvalue = stats.shapiro(x=notnull_col)
    except:
        sw_stat, sw_pvalue = (np.nan, np.nan) 
    try:
        ad_stat, ad_pvalue = diagnostic.normal_ad(x=notnull_col)
    except:
        ad_stat, ad_pvalue = (np.nan, np.nan)
    try:
        ll_stat, ll_pvalue = diagnostic.lilliefors(x=notnull_col, dist='norm')
    except:
        ll_stat, ll_pvalue = (np.nan, np.nan)  
    try:
        dp_stat, dp_pvalue = stats.normaltest(a=notnull_col)
    except:
        dp_stat, dp_pvalue = (np.nan, np.nan)
    try:
        jb_stat, jb_pvalue = stats.jarque_bera(x=notnull_col)
    except:
        jb_stat, jb_pvalue = (np.nan, np.nan) 
    try:
        cp_stat, cp_pvalue = stats.combine_pvalues(pvalues=[sw_pvalue, ad_pvalue, ll_pvalue, dp_pvalue, jb_pvalue], method='fisher')
    except:
        cp_stat, cp_pvalue = (np.nan, np.nan)

    normal_tests = {}
    normal_tests = {'Valores P.': [sw_pvalue, ad_pvalue, ll_pvalue, dp_pvalue, jb_pvalue, cp_pvalue]}
    one_dim_normal_tests = pd.DataFrame(data=normal_tests,
                                        dtype=float,
                                        index=['Shapiro-Wilk (SW)','Anderson-Darling (AD)','Lilliefors (LL)','D’Agostino-Pearson (DP)','Jarque-Bera (JB)','Fisher’s Comb. (FC)'])
    one_dim_normal_tests.index.names = ['Prueba (Test)']
    display(HTML("<h3>Pruebas de normalidad</h3><br>"))
    display(widgets.HTML(one_dim_normal_tests.style\
                        .format({'Valores P.':'{:,.2%}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html() ))

def univar_donut_plot(col):
    col_counts = col.value_counts(ascending=False)
    top_10_counts = col_counts.head(10)
    plt.style.use('default')
    plt.figure(figsize=(5.2, 3.6))
    explode=np.repeat(0.03, len(top_10_counts))
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.pie(x=top_10_counts, explode=explode, labels=None, autopct='%1.1f%%', textprops={'fontsize': 10}, pctdistance=0.85, startangle=90)
    plt.gcf().gca().add_artist(centre_circle)
    plt.title('Diagrama circular - Top 10 categorias', fontweight='bold', fontsize=13)
    leg=plt.legend(labels=top_10_counts.index, loc="upper right", bbox_to_anchor=(3.04, 1), title=col.name, framealpha=0.4)
    plt.setp(leg.get_title(), fontweight='bold', fontsize=10)
    plt.setp(leg.get_texts(), fontsize=9)
    plt.axis('equal')
    plt.show()

def univar_pareto_plot(col):
    col_counts = col.value_counts(ascending=False)
    x_total = col_counts.index
    top_5_counts = col_counts.head(10)
    x = top_5_counts.index
    y = top_5_counts.values
    xlabel=col.name
    ylabel='Frecuencia'
    weights = top_5_counts / col_counts.sum()
    cumsum = weights.cumsum()
    plt.style.use('default')
    fig, ax1 = plt.subplots(figsize=(5.0, 3.4))
    sns.barplot(x=x, y=y, hue=x, palette="bright")
    plt.title('Diagrama de pareto - Top 10 categorias', fontweight='bold', fontsize=13)
    ax1.set_xlabel(xlabel, fontweight='bold', fontsize=11)
    ax1.set_ylabel(ylabel, fontweight='bold', fontsize=11)
    ax1.set_yticklabels(['{:,.0f}'.format(x) for x in ax1.get_yticks()], fontsize=10)
    ax1.legend(loc='right', framealpha=0.4)
    ax2 = ax1.twinx()
    ax2.plot(x, cumsum, '-ro', alpha=0.8, color='gray')
    ax2.set_ylabel('Frec. Relat. Acum.', fontweight='bold', fontsize=11)
    ax2.set_yticklabels(['{:,.0%}'.format(x) for x in ax2.get_yticks()], fontsize=10)
    formatted_weights = ['{:,.0%}'.format(x) for x in cumsum]
    for i, txt in enumerate(formatted_weights):
        ax2.annotate(txt, (x[i], cumsum[i]), fontsize=9, fontweight='bold')
    plt.setp(ax1.legend_.get_texts(), fontsize=9)
    plt.xticks([])
    plt.tight_layout()
    plt.show()

def print_one_way_table(col):       
    if col is None:
        raise TypeError("Verifique que haya asignado algún objeto al parámetro: 'col'")
    elif col.dtype != 'object':
        raise TypeError("Verifique que el valor asignado al parámetros: 'col', es de tipo: 'object'")
    abs_freq = pd.DataFrame(col.value_counts())
    rel_freq = pd.DataFrame(col.value_counts(normalize=True))
    cum_rel_freq = rel_freq.cumsum()
    freq_table = pd.concat([abs_freq, rel_freq, cum_rel_freq], axis=1)
    freq_table.columns = ['Frec.Abs.','Frec.Rel.', 'Frec.Rel.Acum.']
    freq_table.index.name = 'Categorías'
    freq_table.sort_values(by=['Frec.Abs.'], ascending=False)
    display(HTML("<h3>Conteo de frecuencias</h3><br>"))
    display(widgets.HTML(freq_table.style\
                        .format({'Frec.Abs.': '{:,}',
                                 'Frec.Rel.': '{:.2%}',
                                 'Frec.Rel.Acum.': '{:.2%}'})\
                        .set_table_attributes('class="table table-striped"')\
                        .to_html()))


def inter_uncond_descrp_num_var(feat, col):
    tabTab = widgets.Tab()
    outTab = [widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()]
    tabTab.children = outTab
    tabTab.set_title(0, "Frecuencia")
    tabTab.set_title(1, "TC y Posición")
    tabTab.set_title(2, "Dispersión y Forma")
    tabTab.set_title(3, "Normalidad")
    title = 'Estadísticos - ' + feat 
    acTab = widgets.Accordion(children=[tabTab])
    acTab.set_title(0, title)

    with outTab[0]:
        print_one_way_counts_table(col)
    auxOut1 = [widgets.Output(), widgets.Output()]
    with auxOut1[0]:
        print_central_tendency_measures(col)
    with auxOut1[1]:
        print_location_measures(col)
    with outTab[1]:
        display(widgets.HBox(auxOut1))

    auxOut2 = [widgets.Output(), widgets.Output()]
    with auxOut2[0]:
        print_spread_measures(col)
    with auxOut2[1]:
        print_shape_measures(col)
    with outTab[2]:
        display(widgets.HBox(auxOut2))

    with outTab[3]:
        print_normality_tests(col)
        
    tabGraf = widgets.Tab()
    outGraf = [widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output(), widgets.Output()]
    tabGraf.children = outGraf
    tabGraf.set_title(0, "Histograma")
    tabGraf.set_title(1, "Caja y bigotes")
    tabGraf.set_title(2, "Violín")
    tabGraf.set_title(3, "Densidad Norm.")
    tabGraf.set_title(4, "QQ Norm.")

    acGraf = widgets.Accordion(children=[tabGraf])
    title = 'Gráficos descriptivos - ' + feat 
    acGraf.set_title(0, title)

    with outGraf[0]:
        univar_histogram_plot(col)
    with outGraf[1]:
        univar_box_plot(col)
    with outGraf[2]:
        univar_violin_plot(col)
    with outGraf[3]:
        univar_normal_density(col)
    with outGraf[4]:
        univar_qq_norm(col)
        
    out = widgets.Output()  
    with out:
        display(acTab)
        display(acGraf)
    display(out)



def inter_uncond_descrp_cat_var(feat, col):

    tabTab = widgets.Tab()
    outStat = [widgets.Output()]
    tabTab.children = outStat
    tabTab.set_title(0, "Frecuencia")

    title = 'Estadísticos - ' + feat 
    actTab = widgets.Accordion(children=[tabTab])
    actTab.set_title(0, title)

    auxOut1 = [widgets.Output(), widgets.Output()]
    with auxOut1[0]:
        print_one_way_counts_table(col)
    with auxOut1[1]:
        print_one_way_table(col)
    with outStat[0]:
        display(widgets.HBox(auxOut1, layout=Layout(height='300px', overflow='auto')))

    tabGraf = widgets.Tab()
    outGraf = [widgets.Output(), widgets.Output()]
    tabGraf.children = outGraf
    tabGraf.set_title(0, "Circular")
    tabGraf.set_title(1, "Pareto")

    title = 'Gráficos descriptivos - ' + feat 
    actGraf = widgets.Accordion(children=[tabGraf])
    actGraf.set_title(0, title)

    with outGraf[0]:
        univar_donut_plot(col)
    with outGraf[1]:
        univar_pareto_plot(col)

    out = widgets.Output()  
    with out:
        display(actTab)
        display(actGraf)
    display(out)
                      


def corr_pearson(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,        
        fmt=".2f",         
        cmap="viridis",   
        cbar=True          
    )
    plt.show()


def Corr_Kendall(df):  
    correlation_matrix = df.corr(method='kendall')
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,        
        fmt=".2f",         
        cmap="viridis",   
        cbar=True          
    )
    plt.show()

def Corr_Spearman(df):
    correlation_matrix = pd.DataFrame(sta.spearmanr(df).correlation, columns= df.columns, index=df.columns)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        correlation_matrix,
        annot=True,        
        fmt=".2f",         
        cmap="viridis",   
        cbar=True          
    )
    plt.show()


def print_corr_var(df):
    tabTab = widgets.Tab()
    outTab = [widgets.Output(), widgets.Output(), widgets.Output()]
    tabTab.children = outTab
    tabTab.set_title(0, "Pearson")
    tabTab.set_title(1, "Spearman")
    tabTab.set_title(2, "Kendall")


    acTab = widgets.Accordion(children=[tabTab])
    acTab.set_title(0, 'Matriz correlación')

    with outTab[0]: 
        corr_pearson(df)		
    with outTab[1]:
        Corr_Spearman(df)
    with outTab[2]:
        Corr_Kendall(df)

    out = widgets.Output()  
    with out:
        display(acTab)
    display(out)