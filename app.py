import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
import seaborn as sns
import gradio as gr
import numpy as np
import joblib

missing_values = ["--"]
data = pd.read_csv('./macau_weather.csv', na_values = missing_values)
del data['num']
before_rows = data.shape[0]
data = data.dropna()
after_rows = data.shape[0]
adp = after_rows/before_rows*100
edp = (before_rows - after_rows)/before_rows *100

def make_ad_plot():    
    fig = plt.figure()
    plt.title("Percentage of available data")
    data = [adp, edp]
    explode = [0.2,0.2]
    colors = sns.color_palette("Paired")
    plt.pie(data, colors = colors, autopct = '%0.0f%%', explode = explode)
    return fig

def make_ra_table(data):
    table_data = pd.DataFrame(columns=['Before Rain Accumulation', 'Convered Rain Accumulation'])
    clean_data = data.copy()
    clean_data['rain_accum'] = (clean_data['rain_accum']>1) *1
    table_data['Before Rain Accumulation'] = data['rain_accum']
    table_data['Convered Rain Accumulation'] = clean_data['rain_accum']
    return table_data, clean_data

def make_rc_plot():
    corr=clean_data.corr()
    fig = plt.figure(figsize=(8,8))
    sns.heatmap(corr,annot=True,cmap='crest',linewidths=0.2)
    return fig

def make_r_plot(libraries, sd,cd,dd,ld,rc,rd,rl,rn,rf):
    sns.set_theme(style="darkgrid")
    colors = sns.color_palette("Paired")
    rfc = RandomForestClassifier(random_state=25, n_estimators=rn, criterion=rc, max_depth=rd, min_samples_leaf=rl,max_features=rf )
    rfc_s = pd.DataFrame(cross_val_score(rfc,X_train,y_train,cv=10),columns=['RandomForest Score'])
    clf = DecisionTreeClassifier(random_state=25,splitter=sd,criterion=cd, max_depth=dd, min_samples_leaf=ld)
    clf_s = pd.DataFrame(cross_val_score(clf,X_train,y_train,cv=10),columns=['DecisioTree Score'])
    total_socre =pd.DataFrame()
    total_socre['RandomForest'] = rfc_s
    total_socre['DecisionTree'] = clf_s
    
    fig = plt.figure(figsize=(10,5))
    for lib in libraries:
        plt.plot(total_socre[lib],  marker = 'o')
    
    plt.legend(['DecisionTree Score', 'RandomForest Score'])

    plt.title("Final Score ")
    plt.ylabel("Socre")
    plt.xlabel("No. Cross Validation")
    return fig

def make_clf_t_plot():
    fig = plt.figure(figsize=(16,8))
    index = np.arange(0, 2 * 0.2, 0.2) * 2.5
    index = index[0:2]
    bar = plt.bar(index, [clf_score,rfc_score], 0.2, label="Testing Score", color="crimson")
    plt.xticks( index, ['DecisioTree Score','RandomForest Score'])
    plt.yticks(np.arange(0, 1, 0.05))
    # plt.grid(True)
    plt.xlabel("Model")
    plt.ylabel("Test score")
    return fig

def download_clf():
    joblib.dump(clf,"dtc_model.m")
    return "./dtc_model.m"

def download_rfc():
    joblib.dump(rfc,"rfc_model.m")
    return "./rfc_model.m"

table_data, clean_data = make_ra_table(data)

morning_features = ['air_pressure', 'aver_tem', 'humidity',
       'sunlight_time', 'wind_direction', 'wind_speed']
feature=clean_data[morning_features].copy()
label = clean_data['rain_accum'].copy()
X_train,X_test,y_train,y_test = train_test_split(feature,label,test_size=0.33,random_state=324)
clf = DecisionTreeClassifier(random_state=25)
rfc = RandomForestClassifier(random_state=25, n_estimators=11)
clf.fit(X_train,y_train)
rfc.fit(X_train,y_train)
clf_score = clf.score(X_test, y_test)
rfc_score = rfc.score(X_test, y_test)
score = pd.DataFrame([[clf_score,rfc_score],['DecisioTree Score','RandomForest Score']],columns=['DecisioTree Score','RandomForest Score'])

if __name__ == '__main__':
    with gr.Blocks() as demo:
        gr.Markdown("""
        ## Data Collection
        We first collect two years (2020-2021) data from [SMG](https://www.smg.gov.mo/zh/subpage/345/embed-path/p/query-weather-c_panel).
        Below table is sample of the data we collocted
        """
        )

        gr.Dataframe(value = data.head(), overflow_row_behaviour='show_ends'),
        gr.Markdown("""
                        ## Data pre-procesing: 
                        """),        
        with gr.Row():
            with gr.Column():
                gr.Markdown("""
                            ### Data Cleaning: 
                            We mark the **"NaN"** data as **"--"** by pandas, 
                            and we find that there is 623 data is avaliable, anther 3 have NaN column.
                            However, it is just 0.3% in whole data, we decide to delete these data
                            """),
                demo.load(fn=make_ad_plot, inputs=None, outputs=gr.Plot(label = "Pie Plot"))
            with gr.Column():
                gr.Markdown("""
                            ### Data Type Conversion: 
                            Because Decision Tree only accept discrete feature, we conver the rain accumulation. If Rain Accumulation > 1, we think the weather had rained, else it dosen't.\\
                            Below talbe is our convered data:
                            """),
                gr.Dataframe(value = table_data.head(9), overflow_row_behaviour='show_ends')

        gr.Markdown("""
                        ### Feature Selection: 
                        We choose the columns which have high correlation conefficient as feature, rain accumulation as label.
                        """),
        with gr.Row():
            demo.load(fn=make_rc_plot, inputs=None, outputs=gr.Plot(label = "Feature correlation conefficient"))
            gr.Dataframe(value = feature.head(10))

        gr.Markdown("""
                        ## Model Training: 
                        We use DecisionTree and RandomForest to train our data
                        ### Adjust Hyper-parameter
                        """),
        with gr.Box():
            libraries = gr.CheckboxGroup(choices=["DecisionTree","RandomForest"], label="Select Model to display", value=["DecisionTree","RandomForest"])
            with gr.Row():
                with gr.Column():
                    gr.Markdown("""#### DecisionTree""")
                    sd = gr.Radio(['best','random'],value="best",label="splitter of DecisionTree")
                    cd=gr.Radio(['gini', 'entropy'],value="entropy",label="criterion of Decisiontree")
                    dd = gr.Slider(label="max_depth of Decisiontree", value=4, minimum=1, maximum=10, step=1)
                    ld = gr.Slider(label="min_samples_leaf of Decisiontree", value=1, minimum=1, maximum=50, step=5)
                with gr.Column():
                        gr.Markdown("""#### RandomForest""")

                        rc=gr.Radio(['gini', 'entropy'],value="entropy",label="criterion of RandomForest")
                        rd = gr.Slider(label="max_depth of RandomForest", value=4, minimum=1, maximum=10, step=1)
                        rl=gr.Slider(label="min_samples_leaf of RandomForest", value=10, minimum=1, maximum=50, step=5)
                        rn = gr.Slider(label="n_estimators of RandomForest", value=11, minimum=5, maximum=15, step=1)
                        rf =gr.Slider(label="max_features of RandomForest", value=20,minimum=5, maximum=30, step=1)
            with gr.Row():
                train = gr.Button(value="Train")
            train.click(fn=make_r_plot, inputs=[libraries,sd,cd,dd,ld,rc,rd,rl,rn,rf], outputs=gr.Plot(label = "Vaildation Score Plot"))
        gr.Markdown("""
                    ## Testing: 
                    There are the final testing scores
                    """)
        with gr.Row():
            demo.load(fn=make_clf_t_plot, inputs=None, outputs=gr.Plot(label = "Final Score"))
        gr.Markdown("""
                    ## Download Model: 
                    """)    
        with gr.Row():
            with gr.Column():
                clf_model = gr.Button(value="Download DecisionTree Model")
                clf_model.click(fn=download_clf, inputs=None, outputs=gr.File(label="DecisionTree Model"))
            with gr.Column():
                rfc_model = gr.Button(value="Download RandomForest Model")
                rfc_model.click(fn=download_rfc, inputs=None, outputs=gr.File(label="RandomForest Model"))
    demo.launch()  