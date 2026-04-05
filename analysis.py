import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud

plt.rcParams['figure.facecolor'] = '#0D0D0F'
plt.rcParams['axes.facecolor']   = '#16161A'
plt.rcParams['text.color']       = '#F0EEE8'
plt.rcParams['axes.labelcolor']  = '#F0EEE8'
plt.rcParams['xtick.color']      = '#7A7880'
plt.rcParams['ytick.color']      = '#7A7880'
plt.rcParams['axes.edgecolor']   = '#2A2A32'
plt.rcParams['grid.color']       = '#2A2A32'
COLORS = ['#F5C842','#6BCF7F','#42B4F5','#F57C42']

# ── DATA ─────────────────────────────────────────────────
train_texts = [
    'road is broken','pothole on the street','road needs repair',
    'street light not working','road damaged after rain','highway has big holes',
    'footpath is broken','speed breaker is damaged','road is flooded',
    'traffic signal not working','bridge is damaged','no streetlight on highway',
    'road marking is not visible','flyover has cracks','road is slippery',
    'sadak toot gayi hai','sadak pe gadda ho gaya hai','road bahut kharab hai',
    'sadak ki repair nahi hui','street light band hai','sadak pe bada gaddha hai',
    'footpath toot gaya','highway pe bahut holes hai','sadak pe paani jama hai',
    'signal kaam nahi kar raha','sadak ka repair kab hoga',
    'road ke khadde se accident ho gaya','pulia toot gayi hai',
    'speed breaker bahut bada hai','sadak pe marking nahi hai',
    'garbage not collected','trash is overflowing','waste lying on street',
    'dustbin is full','garbage smell is very bad','no garbage pickup since days',
    'waste not being removed','garbage truck has not come','litter all over the place',
    'open garbage dump near house','stray dogs near garbage pile','garbage blocking the road',
    'garbage burning in open','plastic waste on footpath','dead animal on road not removed',
    'kachra nahi utha','kachra bahut zyada jama ho gaya hai','dustbin full ho gaya hai',
    'safai nahi ho rahi','kachra gadi nahi aayi','kachra sadak pe pada hai',
    'safai wala nahi aaya','ganda kachra jama ho gaya','badboo aa rahi hai kachra se',
    'nagar palika kachra nahi utha rahi','khula kachraghara ghar ke paas hai',
    'kachra jal raha hai khule mein','bazar mein bahut gandagi hai',
    'dustbin hi nahi hai hamare area mein','drain kachra se bhar gayi hai',
    'no water supply','water pipe is leaking','dirty water coming from tap',
    'water shortage in area','water tank is empty','sewage water on road',
    'water pressure is very low','borewell is not working','no water for 3 days',
    'muddy water from tap','water meter is broken','pipeline burst on main road',
    'water not reaching upper floors','sewage mixing with drinking water',
    'water smells bad from tap','paani nahi aa raha','nal se paani band hai',
    'paani ki pipe toot gayi','ganda paani aa raha hai','paani ka pressure bahut kam hai',
    'paani tank khali hai','2 din se paani nahi aaya','naali ka paani sadak pe aa gaya',
    'paani ka rang peela hai','boring kaam nahi kar rahi',
    'paani mein badboo aa rahi hai','3 din se paani nahi hai',
    'pipe se paani waste ho raha hai','tanker bhi nahi aa raha','paani peene layak nahi hai',
    'no electricity since morning','power cut for 3 hours','transformer is burnt',
    'power outage in colony','electric wire hanging low on road',
    'meter reading is wrong','no power supply since 2 days',
    'electricity comes and goes every hour','no electricity in the whole street',
    'electricity department not responding','electric pole is leaning and dangerous',
    'fuse blown in our area','electric sparks from wire near house',
    'load shedding every day','inverter not charging due to low voltage',
    'bijli nahi aa rahi','light chali gayi subah se',
    'bijli ka bill bahut zyada aaya hai','transformer jal gaya hai',
    'bijli baar baar aa jaati hai','wire sadak pe latkaa hua hai',
    'bijli vibhag ka koi jawab nahi','meter galat reading de raha hai',
    '2 din se bijli nahi hai','bijli ke khambe jhuk gaya hai',
    'hamare area mein andhera hai raat ko','load shedding roz ho rahi hai',
    'bijli pole sadak pe gir gaya','taar toota pada hai koi nahi uthata',
    'emergency mein bijli gul ho gayi',
]
train_labels = ['road']*30 + ['garbage']*30 + ['water']*30 + ['electricity']*30
labels       = ['road','garbage','water','electricity']

print(f"✅ Dataset: {len(train_texts)} samples")

# ── TRAIN TEST SPLIT ──────────────────────────────────────
X_train,X_test,y_train,y_test = train_test_split(
    train_texts, train_labels, test_size=0.2, random_state=42, stratify=train_labels
)
vectorizer  = TfidfVectorizer(ngram_range=(1,2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_vec, y_train)
lr_pred  = lr_model.predict(X_test_vec)
lr_acc   = accuracy_score(y_test, lr_pred)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred  = nb_model.predict(X_test_vec)
nb_acc   = accuracy_score(y_test, nb_pred)

print(f"✅ Logistic Regression: {lr_acc*100:.2f}%")
print(f"✅ Naive Bayes        : {nb_acc*100:.2f}%")

# ── GRAPH 1: DISTRIBUTION ────────────────────────────────
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(12,5))
fig.suptitle('Dataset Distribution', color='#F0EEE8', fontsize=14, fontweight='bold')
cats   = ['Road','Garbage','Water','Electricity']
counts = [30,30,30,30]
bars = ax1.bar(cats,counts,color=COLORS,width=0.5,edgecolor='none')
ax1.set_title('Samples per Category',color='#F0EEE8')
ax1.set_ylim(0,40); ax1.grid(axis='y',alpha=0.3)
for b,c in zip(bars,counts):
    ax1.text(b.get_x()+b.get_width()/2,b.get_height()+0.5,str(c),
             ha='center',color='#F0EEE8',fontweight='bold')
wedges,texts,autotexts = ax2.pie(counts,labels=cats,colors=COLORS,
    autopct='%1.0f%%',startangle=90,
    wedgeprops={'edgecolor':'#0D0D0F','linewidth':2})
for t in texts: t.set_color('#F0EEE8')
for at in autotexts: at.set_color('#0D0D0F'); at.set_fontweight('bold')
ax2.set_title('Category Split',color='#F0EEE8')
plt.tight_layout()
plt.savefig('dataset_distribution.png',dpi=150,bbox_inches='tight',facecolor='#0D0D0F')
plt.show(); print("✅ Saved: dataset_distribution.png")

# ── GRAPH 2: CONFUSION MATRIX ────────────────────────────
fig,axes = plt.subplots(1,2,figsize=(14,5))
fig.suptitle('Confusion Matrix',color='#F0EEE8',fontsize=14,fontweight='bold')
for ax,pred,title in zip(axes,[lr_pred,nb_pred],
                              ['Logistic Regression','Naive Bayes']):
    cm = confusion_matrix(y_test,pred,labels=labels)
    sns.heatmap(cm,annot=True,fmt='d',ax=ax,
        xticklabels=labels,yticklabels=labels,
        cmap='YlOrRd',linewidths=0.5,linecolor='#0D0D0F')
    ax.set_title(title,color='#F0EEE8',fontweight='bold')
    ax.set_xlabel('Predicted',color='#F0EEE8')
    ax.set_ylabel('Actual',color='#F0EEE8')
plt.tight_layout()
plt.savefig('confusion_matrix.png',dpi=150,bbox_inches='tight',facecolor='#0D0D0F')
plt.show(); print("✅ Saved: confusion_matrix.png")

# ── GRAPH 3: MODEL COMPARISON ────────────────────────────
fig,ax = plt.subplots(figsize=(8,5))
models = ['Logistic\nRegression','Naive\nBayes']
accs   = [lr_acc*100,nb_acc*100]
bars   = ax.bar(models,accs,color=['#C8F04A','#42B4F5'],width=0.4,edgecolor='none')
ax.set_ylim(0,115); ax.set_ylabel('Accuracy (%)',color='#F0EEE8')
ax.set_title('Model Accuracy Comparison',color='#F0EEE8',fontweight='bold')
ax.grid(axis='y',alpha=0.2)
for b,a in zip(bars,accs):
    ax.text(b.get_x()+b.get_width()/2,b.get_height()+1,f'{a:.1f}%',
            ha='center',color='#F0EEE8',fontweight='bold',fontsize=13)
winner = 'Logistic Regression' if lr_acc>=nb_acc else 'Naive Bayes'
ax.text(0.5,0.05,f'Winner: {winner}',transform=ax.transAxes,
        ha='center',color='#C8F04A',fontsize=11)
plt.tight_layout()
plt.savefig('model_comparison.png',dpi=150,bbox_inches='tight',facecolor='#0D0D0F')
plt.show(); print("✅ Saved: model_comparison.png")

# ── GRAPH 4: CROSS VALIDATION ────────────────────────────
all_vec = vectorizer.transform(train_texts)
lr_cv   = cross_val_score(LogisticRegression(max_iter=1000),all_vec,train_labels,cv=5)
nb_cv   = cross_val_score(MultinomialNB(),all_vec,train_labels,cv=5)
fig,ax  = plt.subplots(figsize=(10,5))
x = np.arange(5); w = 0.35
ax.bar(x-w/2,lr_cv*100,w,label='Logistic Regression',color='#C8F04A',edgecolor='none')
ax.bar(x+w/2,nb_cv*100,w,label='Naive Bayes',color='#42B4F5',edgecolor='none')
ax.set_xticks(x); ax.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax.set_title('5-Fold Cross Validation',color='#F0EEE8',fontweight='bold')
ax.set_ylim(0,115); ax.grid(axis='y',alpha=0.2)
ax.legend(facecolor='#16161A',edgecolor='#2A2A32',labelcolor='#F0EEE8')
plt.tight_layout()
plt.savefig('cross_validation.png',dpi=150,bbox_inches='tight',facecolor='#0D0D0F')
plt.show(); print("✅ Saved: cross_validation.png")

# ── GRAPH 5: WORD CLOUDS ─────────────────────────────────
cat_texts = {
    'Road'        :' '.join([t for t,l in zip(train_texts,train_labels) if l=='road']),
    'Garbage'     :' '.join([t for t,l in zip(train_texts,train_labels) if l=='garbage']),
    'Water'       :' '.join([t for t,l in zip(train_texts,train_labels) if l=='water']),
    'Electricity' :' '.join([t for t,l in zip(train_texts,train_labels) if l=='electricity']),
}
fig,axes = plt.subplots(2,2,figsize=(14,8))
fig.suptitle('Word Clouds',color='#F0EEE8',fontsize=14,fontweight='bold')
for ax,(cat,text),color in zip(axes.flatten(),cat_texts.items(),COLORS):
    wc = WordCloud(width=500,height=300,background_color='#16161A',
                   colormap='YlOrRd',max_words=30).generate(text)
    ax.imshow(wc,interpolation='bilinear')
    ax.set_title(cat,color=color,fontweight='bold',fontsize=12)
    ax.axis('off')
plt.tight_layout()
plt.savefig('word_clouds.png',dpi=150,bbox_inches='tight',facecolor='#0D0D0F')
plt.show(); print("✅ Saved: word_clouds.png")

# ── SUMMARY ──────────────────────────────────────────────
print('\n' + '='*50)
print('  PROJECT SUMMARY')
print('='*50)
print(f'  Total Samples         : {len(train_texts)}')
print(f'  Categories            : 4')
print(f'  Logistic Regression   : {lr_acc*100:.2f}%')
print(f'  Naive Bayes           : {nb_acc*100:.2f}%')
print(f'  LR Cross Val Mean     : {lr_cv.mean()*100:.2f}%')
print(f'  Best Model            : {winner}')
print('='*50)
print('  Graphs saved in project folder!')
print('='*50)