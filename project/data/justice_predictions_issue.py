import pdb
import urllib
import pandas as pd
import collections
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.corpus import stopwords
import seaborn as sns
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn import cross_validation
from sklearn.metrics import recall_score,precision_score,accuracy_score
#%matplotlib inline


df_facts = pd.read_table('cases_55-15.csv',sep='|')
stop = stopwords.words('english')
df_facts['facts1']=df_facts['facts_of_the_case'].apply(lambda x: [item.lower() for item in x.split() if item.lower() not in stop])
df_facts['facts1']=df_facts['facts1'].str.join(" ")

#vect= CountVectorizer()
#X = vect.fit_transform(df_facts['facts1'].values)


df_justice_all = pd.read_csv('SCDB_2015_01_justiceCentered_LegalProvision.csv')
#df_justice_all.loc[ df_justice_all['caseSource'].isnull(),'caseSource'] = 0

def predict_judge(judge):
	df_justice = df_justice_all[ df_justice_all['justiceName'] == judge ]


	df = pd.merge(df_justice,df_facts,left_on='docket',right_on='docket_number')

	df_term = {'justiceName':['AScalia', 'AMKennedy', 'CThomas','RBGinsburg', 'SGBreyer', 'JGRoberts', 'SAAlito', 'SSotomayor','EKagan'],
             'start_year':[1986,1988,1991,1993,1994,2005,2006,2009,2010]}
	df = pd.merge(pd.DataFrame.from_dict(df_term),df,left_on='justiceName',right_on='justiceName')
	df['year_of_service'] = df['year'] - df['start_year']
#train_cols = ['caseOriginState','issue','issueArea','lcDispositionDirection','caseSource','caseSourceState']
	train_cols = ['issue','issueArea','lcDispositionDirection','petitioner','respondent']

	df_with_labels = df[ df['direction'].isin([1,2]) ]


#df_with_labels = df_with_labels[train_cols+['direction']]
#	df_with_labels = df_with_labels[train_cols+['year_of_service','direction']]
	df_with_labels = df_with_labels[train_cols+['direction']]

#	print len(df_with_labels)
#	print df_with_labels[ df_with_labels.isnull().any(axis=1) ]
	df_with_labels = df_with_labels[ df_with_labels.notnull().all(axis=1) ].reset_index()


	df_binarized = pd.concat([pd.get_dummies(df_with_labels[col]) for col in train_cols], axis=1)
#df_binarized['year_of_service'] = df_with_labels['year_of_service']
	df_binarized['direction'] = df_with_labels['direction']

#pdb.set_trace()
#df_binarized = df_binarized[ df_binarized.notnull().all(axis=1) ].reset_index()

	avg_total = 0
	baseline_total = 0
	kf = cross_validation.KFold(len(df_binarized), n_folds=5,shuffle = True)
	model_RF = RandomForestClassifier(n_estimators=80)
#	model_B = BaggingClassifier(model)
	model_KN = KNeighborsClassifier(n_neighbors=10)
	model_SVC = LinearSVC()
	model_Logistic = LogisticRegression()
#	for model,model_name in [(model_RF,"random forest"),(model_SVC,"linear svc"),(model_Logistic,"logistic")]:
	for model,model_name in [(model_RF,"random forest")]:
		avg_total = 0
		baseline_total = 0
		for train_index, test_index in kf:
			train, test = df_binarized.ix[train_index,:], df_binarized.ix[test_index,:].reset_index()
			train = train.reset_index()

		 	model=model.fit( train.ix[:, train.columns != 'direction'], train['direction'] )

                #pred_probs = model.predict_proba(test[train_cols])[::,1]
                #preds = model.predict(test[train_cols])
			preds = model.predict(test.ix[:, test.columns != 'direction'])
#	                print accuracy_score(test['direction'],preds)
			avg_total += accuracy_score(test['direction'],preds)
			test_with_labels = df_with_labels.ix[test_index,:].reset_index()

#                match = df_with_labels.ix[test_index,'direction']!=preds
#                match = match.reset_index(drop=True)
			combined =  pd.concat([test_with_labels,pd.DataFrame( preds,columns=['pred'])],axis=1)
			combined['correct'] = combined['pred']==combined['direction']

			df_issues = {'issueArea':[1,2,3,4,5,6,7,8,9,10,11,12,13,14],
             'issueAreaText':['criminal procedure','civil rights','First Ammendment','due process',
                             'privacy','attorney or gov officials compensation','unions','economic activity',
                              'judicial power','federalism','interstate relation','federal taxation',
                              'misc','private law']}
	        	combined = pd.merge(pd.DataFrame.from_dict(df_issues),combined,left_on='issueArea',right_on='issueArea')
#	                print combined.groupby('issueAreaText')['correct'].mean()
#                print test_with_labels['issueArea'].value_counts() / len(test_with_labels)
	        	mislabeled =  combined[ combined['pred']!=combined['direction']  ]
#                print mislabeled['issueArea'].value_counts() / len(mislabeled)

			combined_by_issue = combined.groupby('issueAreaText')
			for name,group in combined_by_issue:
				cts= group['direction'].value_counts()#.max / len(group)
				print len(group)
				print cts.max()/float(len(group))
#			print combined_by_issue['direction'].value_counts()
			print combined_by_issue['correct'].mean()
			curr_baseline = test['direction'].value_counts()/len(test)
			curr_baseline = curr_baseline.max()
			baseline_total+=curr_baseline
			break
		print judge




for j in ['AScalia', 'AMKennedy', 'CThomas','RBGinsburg', 'SGBreyer', 'JGRoberts', 'SAAlito', 'SSotomayor','EKagan']:
	predict_judge(j)
	print '--------------------------'

