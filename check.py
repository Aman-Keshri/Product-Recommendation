import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask,render_template, request


def get_data():
	df1 = pd.read_csv('product_detail.csv')
	df1.drop(['image','style'],axis=1,inplace=True)

	def convert_vote(x):
	    try:
	        x = float(x)
	    except Exception:
	        return float('nan')

	df1['vote']=df1['vote'].apply(convert_vote)

	df1.fillna(0.0,inplace=True)
	df1 = df1.applymap(lambda x: 1 if x == True else x)
	df1 = df1.applymap(lambda x: 0 if x == False else x)
	df1.drop(['reviewText','reviewTime','summary','unixReviewTime','vote','verified','reviewerName'],axis=1,inplace=True)

	return df1


def rcmd(user):

	df = get_data()

	rate = df.filter(['asin', 'overall', 'reviewerID'])
	Mean = rate.groupby(by="reviewerID",as_index=False)['overall'].mean()
	rating_avg = pd.merge(rate,Mean,on='reviewerID')
	rating_avg['adg_rating']=rating_avg['overall_x']-rating_avg['overall_y']
	product = df.filter(['asin','product_name'])

	product_matrix=pd.pivot_table(rating_avg,index='reviewerID',columns='asin',values='overall_x')
	user_matrix=pd.pivot_table(rating_avg,index='reviewerID',columns='asin',values='adg_rating')

	final_product = user_matrix.fillna(user_matrix.mean(axis=0))
	final_user = user_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

	b = cosine_similarity(final_user)
	np.fill_diagonal(b, 0 )
	similarity_with_user = pd.DataFrame(b,index=final_user.index)
	similarity_with_user.columns=final_user.index

	cosine = cosine_similarity(final_product)
	np.fill_diagonal(cosine, 0 )
	similarity_with_product = pd.DataFrame(cosine,index=final_product.index)
	similarity_with_product.columns=final_product.index

	def find_n_neighbours(df,n):  
	    df = df.apply(lambda x: pd.Series(x.sort_values(ascending=False)
	           .iloc[:n].index, 
	          index=['top{}'.format(i) for i in range(1, n+1)]), axis=1)
	    return df

	sim_user_30_u = find_n_neighbours(similarity_with_user,30)
	sim_user_30_m = find_n_neighbours(similarity_with_product,30)

	Rating_avg = rating_avg.astype({"asin": str})
	product_user =rating_avg.groupby(by = 'reviewerID')['asin'].apply(lambda x:','.join(x))

	def User_item_score(user):
	    product_seen_by_user = product_matrix.columns[product_matrix[product_matrix.index==user].notna().any()].tolist()
	    a = sim_user_30_m[sim_user_30_m.index==user].values
	    b = a.squeeze().tolist()
	    d = product_user[product_user.index.isin(b)]
	    l = ','.join(d.values)
	    product_seen_by_similar_users = l.split(',')
	    product_under_consideration = list(set(product_seen_by_similar_users)-set(list(map(str, product_seen_by_user))))
	    product_under_consideration = list(map(str, product_under_consideration))
	    score = []
	    for item in product_under_consideration:
	        c = final_product.loc[:,item]
	        d = c[c.index.isin(b)]
	        f = d[d.notnull()]
	        avg_user = Mean.loc[Mean['reviewerID'] == user,'overall'].values[0]
	        index = f.index.values.squeeze().tolist()
	        corr = similarity_with_user.loc[user,index]
	        fin = pd.concat([f, corr], axis=1)
	        fin.columns = ['adg_score','correlation']
	        fin['score']=fin.apply(lambda x:x['adg_score'] * x['correlation'],axis=1)
	        nume = fin['score'].sum()
	        deno = fin['correlation'].sum()
	        final_score = avg_user + (nume/deno)
	        score.append(final_score)
	    data = pd.DataFrame({'asin':product_under_consideration,'score':score})
	    top_10_recommendation = data.sort_values(by='score',ascending=False).head(10)
	    #apple = pd.DataFrame({'asin':['B0006GVNOA']})
	    #top_10_recommendation = pd.concat([apple, top_10_recommendation]).reset_index(drop = True)
	    product_Name = top_10_recommendation.merge(product, how='inner', on='asin')
	    product_Names = product_Name.product_name.values.tolist()
	    return product_Names

	try:
		predicted_product = User_item_score(user)
	except (KeyError, IndexError):
	    predicted_product = User_item_score('A1AHW6I678O6F2')
	else:
	    predicted_product = User_item_score(user)
	    
	def unique(predicted_product): 
	    unique_list = [] 
	    for x in predicted_product:  
	        if x not in unique_list: 
	            unique_list.append(x) 
	    return unique_list
	
	final_list=unique(predicted_product)
	    
	return final_list


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('login.html')


@app.route('/',methods=['POST','GET'])
def predict():
    user_ID = request.args.get('user_ID')
    final_prediction = rcmd(user_ID)
    return render_template('login.html', prediction=final_prediction)


if __name__ == "__main__":
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True,host='127.0.0.1')
