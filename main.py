from cgi import test
from gettext import install
from matplotlib import image
from PIL import Image
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import keras
from keras.applications.imagenet_utils import decode_predictions
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression
import warnings
from lime import lime_image
from skimage.segmentation import mark_boundaries
import lime
from lime import lime_tabular
from sklearn.model_selection import train_test_split
from streamlit import components
from lime.lime_text import LimeTextExplainer
from keras.models import load_model
import sklearn.ensemble
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
from keras.applications.inception_v3 import InceptionV3
import string
from PIL import Image
import jieba
    
# 設定網頁標題
product_list= ["影像數據","結構化數據","文本數據"]
product_type=st.sidebar.selectbox(
	"選擇模型的類型",
	 product_list
)
if product_type=="影像數據":
    perturb = st.sidebar.slider('num_perturb',150,1000)
    st.title('可解釋機器學習(影像數據)')
if product_type=="結構化數據":
    st.title('可解釋機器學習(結構化數據)')
if product_type=="文本數據":
    st.title('可解釋機器學習(文本數據)')    
def plot_preds(image, preds,labels):

    plt.imshow(image)
    plt.axis('off')
    plt.figure()
    plt.barh([0, 1,2,3,4], preds, alpha=0.5)
    plt.yticks([0, 1,2,3,4], labels)
    plt.xlabel('Probability')
    plt.xlim(0,1.01)
    plt.tight_layout()
    plt.show()

if product_type=="影像數據":
    def perturb_image(img,perturbation,segments):
        active_pixels = np.where(perturbation == 1)[0]
        mask = np.zeros(segments.shape)
        for active in active_pixels:
            mask[segments == active] = 1 
        perturbed_image = copy.deepcopy(img)
        perturbed_image = perturbed_image*mask[:,:,np.newaxis]
        return perturbed_image
        
    img=st.file_uploader("上傳一張要比較的圖片",type="jpg")
    if img:
        st.image(img, caption='你的圖片')
        img = skimage.io.imread(img)
        h=img.shape[0]
        w=img.shape[1]
        img = skimage.transform.resize(img, (299,299)) 
        img = (img - 0.5)*2 
    

    
    if st.button("run"):
        warnings.filterwarnings('ignore') 
        inceptionV3_model =  keras.applications.inception_v3.InceptionV3()
        np.random.seed(222)
        preds = inceptionV3_model.predict(img[np.newaxis,:,:,:])
        st.write("辨識結果為",(decode_predictions(preds)[0][0][1]),"的機率=",round(float(format((decode_predictions(preds)[0][0][2]),'.5f'))*100,3),"%")
        
        top_pred_classes = preds[0].argsort()[-6:][::-1] #開始解釋
        superpixels = skimage.segmentation.quickshift(img, kernel_size=4,max_dist=200, ratio=0.2)
        num_superpixels = np.unique(superpixels).shape[0]
        num_perturb = perturb
        perturbations = np.random.binomial(1, 0.5, size=(num_perturb, num_superpixels))
        predictions = []
        for pert in perturbations:
            perturbed_img = perturb_image(img,pert,superpixels)
            pred = inceptionV3_model.predict(perturbed_img[np.newaxis,:,:,:])
            predictions.append(pred)

        predictions = np.array(predictions)
        original_image = np.ones(num_superpixels)[np.newaxis,:] #Perturbation with all superpixels enabled 
        distances = sklearn.metrics.pairwise_distances(perturbations,original_image, metric='cosine').ravel()
        
        kernel_width = 0.25
        weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
        
        class_to_explain = top_pred_classes[0]
        simpler_model = LinearRegression()
        simpler_model.fit(X=perturbations, y=predictions[:,:,class_to_explain], sample_weight=weights)
        coeff = simpler_model.coef_[0]
        intercept =simpler_model.intercept_[0]
        num_top_features = 10
        top_features = np.argsort(coeff)[-num_top_features:] 
        mask = np.zeros(num_superpixels) 
        mask[top_features]= True #Activate top superpixels
        img=perturb_image(img/2+0.5,mask,superpixels)
        img = skimage.transform.resize(img, (h,w))
        st.image(img,caption="解釋結果")
        
        
if product_type=="結構化數據":
    col1, col2, col3 = st.columns(3)
    income = col1.number_input("收入:")
    married = col2.selectbox("是否結婚?",["YES","NO"])
    region = col3.selectbox("地區:",["INNER_CITY", "TOWN", "SUBURBAN", "RURAL"])
    save_act = col2.selectbox("是否有活儲帳戶?",["YES","NO"])
    children = col3.selectbox("小孩個數?",[0,1,2,3])
    df_pred = pd.DataFrame([[region,income,married,children,save_act]],
                       columns= ['region','income','married','children','save_act'])
    df_pred.replace({'YES':1, 'NO':0, 'M': 1, 'F': 0}, inplace=True)
    df_pred = pd.get_dummies(df_pred, prefix=['region','children'], prefix_sep='_', columns=['region','children'])
    model = joblib.load('small_loan2_rf_model.pkl')
    missing_cols = [c for c in model.feature_names_in_ if c not in df_pred.columns]
    for c in missing_cols:
        df_pred[c] = 0
    df_pred = df_pred[model.feature_names_in_]
    small_loan = pd.read_csv('small_loan2.csv')
    small_loan.replace({'YES':1, 'NO':0, 'M': 1, 'F': 0}, inplace=True)
    small_loan = pd.get_dummies(small_loan, prefix=['region','children'], prefix_sep='_', columns=['region','children'])
    X = small_loan.drop(['id','response'],axis=1)
    X_train, X_test= train_test_split(X, test_size=0.2, random_state=107)
    prediction = model.predict(df_pred)
    prediction_prob = model.predict_proba(df_pred)
    
    
    if st.button("run"):
        if(prediction[0]==0):
            st.write('<p class="big-font">此人對小額信貸<font color="red">沒有興趣</font>.</p>',unsafe_allow_html=True) 
        else:
            st.write('<p class="big-font">此人對小額信貸<font color="blue">有興趣</font>.</p>',unsafe_allow_html=True)  
        explainer = lime_tabular.LimeTabularExplainer(training_data=np.array(X_train),class_names=['Not interested in','interested in'],feature_names=X_train.columns)
        exp = explainer.explain_instance(df_pred.iloc[0], model.predict_proba,num_features=11)   
        html_lime = exp.as_html()
        st.subheader('Lime Explanation')
        components.v1.html(html_lime, width=1000, height=500, scrolling=True)
        
if product_type=="文本數據":
    input_text = st.text_input('Enter your text:', "")
    data=pd.read_csv('output.csv')
    f=open(r'stop_words.txt','r',encoding='utf-8')
    stop_words = f.readlines()
    f.close()
    for i in range(len(stop_words)):
        stop_words[i]=stop_words[i].replace('\n','')
    x=data.Seg_Text
    y=data.Target
    x_train_data, x_test_data, y_train_data, y_test_data = train_test_split(x, y, test_size=0.2, random_state=107)
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(lowercase=False)
    #下面兩個步驟為轉成類別型資料
    model = joblib.load('test_model.pkl') 
    c = make_pipeline(vectorizer, model)
    
    vectorizer.fit_transform(x_train_data)
    
    if st.button("run"):
        input_text1=[]
        documents=[input_text]
        for sentence in documents:
            seg_list = jieba.cut(sentence)
            input_text1.append(' '.join(seg_list))
        for j in range(len(stop_words)):
            input_text1[0]=input_text1[0].replace(stop_words[j],'')
        explainer = LimeTextExplainer(class_names=["負面的情緒","正面的情緒"])
        s=input_text1[0]
        exp = explainer.explain_instance(s, c.predict_proba, num_features=10)
        if((c.predict_proba(["分析不出來"])[0,1])==(c.predict_proba([s])[0, 1])):
            st.write('<p class="big-font">此段文字<font color="red">無法判斷</font></p>',unsafe_allow_html=True)
        elif((c.predict_proba([s])[0, 1])>0.5):
            st.write('<p class="big-font">此段文字為<b><font color="#FF8E4D">正面的情緒</font></b></p>',unsafe_allow_html=True)
            html_lime = exp.as_html()     
            st.subheader('Lime Explanation')
            components.v1.html(html_lime, width=800, height=500, scrolling=True)
        else:
            st.write('<p class="big-font">此段文字為<font color="blue">負面的情緒</font></p>',unsafe_allow_html=True)
            html_lime = exp.as_html()     
            st.subheader('Lime Explanation')
            components.v1.html(html_lime, width=800, height=500, scrolling=True)
        