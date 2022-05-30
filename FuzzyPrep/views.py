from django.shortcuts import render
from rest_framework import serializers
from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from importlib import import_module
from django.conf import settings
import pandas as pd
import numpy as np
import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import unicodedata as ud
import scipy
from pyjarowinkler import distance
import sklearn.preprocessing as pp
import os
import csv
from sparse_dot_topn import awesome_cossim_topn
import sys
import json
# import psutil

from .forms import UploadFileForm, GetFiles
from .models import FileForm

def index(request):
    form = UploadFileForm()
    return render(request, 'fuzzyprep/index.html', {'form': form})

def upload(request):
    if request.method == 'POST':
        file_top = request.FILES['topfile']
        file_bottom = request.FILES['bottomfile']
        if (str(file_top).endswith(".xlsx")) :
            df_top = pd.read_excel(request.FILES['topfile'])
        elif str(file_top).endswith(".csv") | str(file_top).endswith("CSV"):
            df_top = pd.read_csv(request.FILES['topfile'])
        
        if(str(file_bottom).endswith(".csv") | str(file_bottom).endswith("CSV")):
            df_bottom = pd.read_csv(request.FILES['bottomfile'])
        elif str(file_bottom).endswith("xlsx") :
            df_bottom = pd.read_excel(request.FILES['bottomfile'])

        up_col_name = request.POST['topColname']
        down_col_name = request.POST['bottomColname']
        result = fuzzymatching(df_top[up_col_name],up_col_name, df_bottom[down_col_name],down_col_name, request.POST['matchstyle'])
         
        return render(request, 'fuzzyprep/select.html', {'data_top': result[up_col_name+'_up'][1:10],'data_bottom': result[down_col_name+'_down'][1:10], 'score': result['scores'][1:10], 'jw_scores': result['jw_scores'][1:10], 'token': result['token_analysis_result'][1:10]})
    else: form = UploadFileForm()
    return render(request, 'fuzzyprep/index.html', {'form': form})

def selectData(request):
    return render(request, 'fuzzyprep/select.html')

def getAvailableMemorySize():
    size = psutil.virtual_memory().available
    return JsonResponse(data=responseSerializer({'response': size}).data, safe=False)


class responseSerializer(serializers.Serializer):
    response=serializers.JSONField()

@csrf_exempt 
def getSheetnames(request):
    if request.method == 'POST':
        request.session['file1'] = request.FILES['file1']
        request.session['file2'] = request.FILES['file2']

        request.session.create()

        file1 = request.FILES['file1']
        if (str(file1).endswith("xlsx")) :
            df = pd.ExcelFile(file1)
            result1=[]
            for name in df.sheet_names:
                result1.append(name)
            
        file2 = request.FILES['file2']
        if str(file2).endswith("xlsx"):
            df = pd.ExcelFile(file2)
            result2=[]
            for name in df.sheet_names:
                result2.append(name)
        if ((str(file1).endswith("xlsx")) & (str(file2).endswith("xlsx"))):
            result = {"file1": result1,"file2": result2, "sessionid": request.session.session_key}
        elif str(file1).endswith("xlsx"):
            result = {"file1": result1, "sessionid": request.session.session_key}
        else:
            result = {"file2": result2, "sessionid": request.session.session_key}
        
        return JsonResponse(data=responseSerializer({'response':result}).data, safe=False)
    return JsonResponse(data=responseSerializer({'response': 'Something went wrong during file reading process, please try again.'}).data, safe=False)

@csrf_exempt
def getColumnNames(request):
    SessionStore = import_module(settings.SESSION_ENGINE).SessionStore
    if request.method == 'POST':
        try:
            request.FILES['file1']
            request.FILES['file2']
        except:
            s = SessionStore(session_key=request.POST['sessionid'])
            s['selection1'] = request.POST['selection1']
            s['selection2'] = request.POST['selection2']
            file1 = s['file1']
            file2 = s['file2']
            new_session = False
        else:
            file1 = request.FILES['file1']
            file2 = request.FILES['file2']
            request.session['file1'] = request.FILES['file1']
            request.session['file2'] = request.FILES['file2']
            request.session['selection1'] = request.POST['selection1']
            request.session['selection2'] = request.POST['selection2']
            new_session = True
        
        if (str(file1).endswith(".xlsx")) :
            df_top = pd.read_excel(file1,sheet_name=request.POST['selection1'])
        elif str(file1).endswith(".csv") | str(file1).endswith("CSV"):
            df_top = pd.read_csv(file1,sep=request.POST['selection1'])
        
        if str(file2).endswith("xlsx"):
            df_bottom = pd.read_excel(file2,sheet_name=request.POST['selection2'])
        elif (str(file2).endswith(".csv") | str(file2).endswith("CSV")) :
            df_bottom = pd.read_csv(file2, sep=request.POST['selection2'])

        if new_session:
            request.session['df_top'] = df_top
            request.session['df_bottom'] = df_bottom
            request.session.create()
            result = {'cfile1': list(df_top.columns), 'cfile2': list(df_bottom.columns),  "sessionid": request.session.session_key}
            return JsonResponse(data=responseSerializer({'response': result}).data, safe=False)
        else:
            s['df_top'] = df_top
            s['df_bottom'] = df_bottom
            s.save()
            result = {'cfile1': list(df_top.columns), 'cfile2': list(df_bottom.columns)}
            return JsonResponse(data=responseSerializer({'response': result}).data, safe=False)

@csrf_exempt
def doMatching(request):
    if request.method == 'POST':
        SessionStore = import_module(settings.SESSION_ENGINE).SessionStore
        s = SessionStore(session_key=request.POST['sessionid'])
        df_top = s['df_top']
        df_bottom = s['df_bottom']

        i = 0
        result=pd.DataFrame()
        styles = request.POST['style'].split(',')
        nonex = json.loads(request.POST['nonex'])
        cases = request.POST['cases'].split(',')
        ngrams = request.POST['ngrams'].split(',')
        thresh = request.POST['thresh'].split(',')

        for col1 in request.POST['columns1'].split(','):
            col2 = request.POST['columns2'].split(',')[i]
            # matchResult = fuzzymatching(df_top.loc[:,[col1]], df_bottom.loc[:,[col2]], request.POST['style'])
            if cases[i] == 'case_insensitive':
                lower_case = True
            else: 
                lower_case = False
            if styles[i] == 'MJS':
                matchResult = FuzzyMatching_MJS(df_top.loc[:,[col1]],df_bottom.loc[:,[col2]],non_exclusion=nonex[i]['nonexvalue'],lower_case=lower_case,top_n_result =int(ngrams[i]),min_thresh =float(thresh[i]))
            if styles[i] == 'CSS':
                matchResult = FuzzyMatching_CSS(df_top.loc[:,[col1]],df_bottom.loc[:,[col2]], nonex[i]['nonexvalue'],lower_case, int(ngrams[i]), float(thresh[i]))
            mergeResult = pd.merge(df_top, matchResult, left_on=df_top.loc[:,[col1]].columns[0], right_on=matchResult.columns[0], how='right')
            mergeResult = pd.merge(mergeResult, df_bottom, left_on=matchResult.columns[1], right_on=df_bottom.loc[:,[col2]].columns[0], how='left')
            mergeResult.drop([matchResult.columns[0], matchResult.columns[1]], axis=1, inplace=True)
            mergeResult['match_level'] = i
            result = result.append(mergeResult)
            i = i+1
        
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename=filename.csv'
        result.to_csv(path_or_buf=response,float_format='%.2f',index=False,decimal=",")
        
        # os.remove(os.path.join(os.path.dirname(os.path.abspath(__file__)),'files\sessionid'+request.POST['sessionid']))

        if not response.has_header('Access-Control-Allow-Origin'):
            response['Access-Control-Allow-Origin']  = '*'
        
        if response.status_code != 200:
            if response.status_code == 500:
                return JsonResponse(data=responseSerializer({'response': 'Something went wrong within backend server'}).data, safe=False)
            return JsonResponse(data=responseSerializer({'response': 'Something went wrong'}).data, safe=False)
        
        return response

    return JsonResponse(data=responseSerializer({'response': 'Something went wrong during matching, please try again.'}).data, safe=False)        

def FuzzyMatching_MJS(up_slice,down_slice,non_exclusion='',lower_case=True,top_n_result =3,min_thresh =0.70):
    # Function Start here!
    up_col_name = up_slice.columns[0]
    down_col_name = down_slice.columns[0]
    up_slice1 = up_slice[up_col_name]
    down_slice1 = down_slice[down_col_name]

    #dimension reduction
    up_slice1.drop_duplicates( keep="first", inplace=True)
    down_slice1.drop_duplicates( keep="first", inplace=True)
    up_slice1.replace('', np.nan, inplace=True)
    down_slice1.replace('', np.nan, inplace=True)
    up_slice1.dropna(inplace=True)
    down_slice1.dropna(inplace=True)

    #preliminary cleaning
    up=up_slice1.apply(lambda x : ud.normalize('NFKD', x).encode('ascii', 'ignore').decode("utf-8")   )
    down=down_slice1.apply(lambda x : ud.normalize('NFKD', x).encode('ascii', 'ignore').decode("utf-8")  )
    if '-' in non_exclusion and non_exclusion[0]!='-':
        non_exclusion=non_exclusion.replace('-','')
        non_exclusion = '-'+non_exclusion
    up=up.apply(lambda x :  re.sub('\s+', ' ', re.sub(r'[^a-zA-Z0-9 '+non_exclusion+']', r'', x)).strip()    )
    down=down.apply(lambda x :re.sub('\s+', ' ', re.sub(r'[^a-zA-Z0-9 '+non_exclusion+']', r'', x)).strip()    )
    up = list(up)
    down = list(down)
    full=up+down

    #sparse matrix creation
    vectorizer = TfidfVectorizer(lowercase=lower_case,analyzer='word',dtype=np.float64)
    tf_idf_matrix = vectorizer.fit_transform(full)

    row_index = tf_idf_matrix.nonzero()[0]
    col_index = tf_idf_matrix.nonzero()[1]
    #binary matrix creation
    bi_data = np.array([1]*tf_idf_matrix.getnnz())
    tf_idf_matrix_binary = csr_matrix((bi_data, (row_index, col_index)))
    #power matrix creation
    tf_idf_matrix_pow = tf_idf_matrix.power(n=2)
 
    len_up = len(up)
    len_down = len(down)
    #slice und dice
    tf_idf_matrix_pow_up = tf_idf_matrix_pow[0:len_up,:].astype(np.float64)
    tf_idf_matrix_pow_down = tf_idf_matrix_pow[len_up: ,:].astype(np.float64)
    tf_idf_matrix_binary_up_T = tf_idf_matrix_binary[0:len_up,:].T.astype(np.float64)
    tf_idf_matrix_binary_down_T = tf_idf_matrix_binary[len_up: ,:].T.astype(np.float64)

    #constant memory usage bytes
    m = 0.8*1024*1024*1024
    len_up_max_chunk_size = int(m/len_down)

    result_up = []
    result_down = []
    result_score = []
    #for up_to_down dir analysis
    for i in range(0,len_up,len_up_max_chunk_size):
        if len_up-i<len_up_max_chunk_size:
            bound = len_up-i
        else:
            bound = len_up_max_chunk_size
        
        result=awesome_cossim_topn(tf_idf_matrix_pow_up[i:i+bound,:], tf_idf_matrix_binary_down_T,
                              ntop= top_n_result, lower_bound=min_thresh , use_threads=True, n_jobs=4)

        result_up+=  [    up_slice1[idx] for idx in   (result.nonzero()[0]+i)  ]  
        result_down+=  [  down_slice1[idx] for idx in result.nonzero()[1] ]  
        result_score+= list(result.data)

    
    result_df = pd.DataFrame(list(zip(result_up, result_down,result_score)),
                      columns=[up_col_name+'_up', down_col_name+'_down','mjs_scores'])

    return result_df

def FuzzyMatching_CSS(up_slice,down_slice,non_exclusion='',lower_case=True,top_n_result =3,min_thresh =0.70):
    # Function Start here!
    up_col_name = up_slice.columns[0]
    down_col_name = down_slice.columns[0]
    up_slice1 = up_slice[up_col_name]
    down_slice1 = down_slice[down_col_name]

    #dimension reduction
    up_slice1.drop_duplicates( keep="first", inplace=True)
    down_slice1.drop_duplicates( keep="first", inplace=True)
    up_slice1.replace('', np.nan, inplace=True)
    down_slice1.replace('', np.nan, inplace=True)
    up_slice1.dropna(inplace=True)
    down_slice1.dropna(inplace=True)

    #preliminary cleaning
    up=up_slice1.apply(lambda x : ud.normalize('NFKD', x).encode('ascii', 'ignore').decode("utf-8")   )
    down=down_slice1.apply(lambda x : ud.normalize('NFKD', x).encode('ascii', 'ignore').decode("utf-8")  )
    if '-' in non_exclusion and non_exclusion[0]!='-':
        non_exclusion=non_exclusion.replace('-','')
        non_exclusion = '-'+non_exclusion
    up=up.apply(lambda x :  re.sub('\s+', ' ', re.sub(r'[^a-zA-Z0-9 '+non_exclusion+']', r'', x)).strip()    )
    down=down.apply(lambda x :re.sub('\s+', ' ', re.sub(r'[^a-zA-Z0-9 '+non_exclusion+']', r'', x)).strip()    )
    up = list(up)
    down = list(down)
    full=up+down

    #sparse matrix creation
    vectorizer = TfidfVectorizer(lowercase=lower_case,analyzer='word',dtype=np.float64)
    tf_idf_matrix = vectorizer.fit_transform(full)

    len_up = len(up)
    len_down = len(down)
    #slice und dice
    tf_idf_matrix_up = tf_idf_matrix[0:len_up,:].astype(np.float64)
    tf_idf_matrix_down_T = tf_idf_matrix[len_up: ,:].T.astype(np.float64)

    #constant memory usage bytes
    m = 0.8*1024*1024*1024
    len_up_max_chunk_size = int(m/len_down)

    result_up = []
    result_down = []
    result_score = []
    #for up_to_down dir analysis
    for i in range(0,len_up,len_up_max_chunk_size):
        if len_up-i<len_up_max_chunk_size:
            bound = len_up-i
        else:
            bound = len_up_max_chunk_size
        
        result=awesome_cossim_topn(tf_idf_matrix_up[i:i+bound,:], tf_idf_matrix_down_T,
                               ntop=top_n_result, lower_bound=min_thresh ,use_threads=True,n_jobs=4)

        result_up+=  [    up_slice1[idx] for idx in   (result.nonzero()[0]+i)  ]  
        result_down+=  [  down_slice1[idx] for idx in result.nonzero()[1] ]  
        result_score+= list(result.data)
        
    result_df = pd.DataFrame(list(zip(result_up, result_down,result_score)),
                      columns=[up_col_name+'_up', down_col_name+'_down','css_scores'])
    return result_df

def fuzzymatching(up_slice, down_slice, style):

    up_col_name = up_slice.columns[0]
    down_col_name = down_slice.columns[0]


    up_slice1 = up_slice[up_col_name]
    down_slice1 = down_slice[down_col_name]

    up_slice1.drop_duplicates( keep="first", inplace=True)
    down_slice1.drop_duplicates( keep="first", inplace=True)

    up_slice1.replace('', np.nan, inplace=True)
    down_slice1.replace('', np.nan, inplace=True)
    up_slice1.dropna(inplace=True)
    down_slice1.dropna(inplace=True)
    up_slice1 = list(up_slice1)
    down_slice1 = list(down_slice1)

    #treating issue if input lists too short to prevent matrix top n error
    temp_list =['abc','bcd','xyz']
    if len(up_slice1)<3:
        up_slice1=up_slice1+temp_list
    if len(down_slice1)<3:
        down_slice1=down_slice1+temp_list

    up = []
    down = []
    for i in up_slice1:
        g = ud.normalize('NFKD', str(i)).encode('ascii', 'ignore')
        f =  clean(  g.decode("utf-8")   )
        up.append(f)
    for i in down_slice1:
        g = ud.normalize('NFKD', str(i)).encode('ascii', 'ignore')
        f =  clean(  g.decode("utf-8")    )
        down.append(f)

    full = up + down

    # 0. Fuzzy Matching   - modified jaccard similarity    MJS
    result_up = []
    result_down = []
    scores = []
    pace = 50
    if style in ['Default','MJS']:
        
        # Matrix creations
        # original whole matrix
        vectorizer = TfidfVectorizer(min_df=1, analyzer=whole_token)
        tf_idf_matrix = vectorizer.fit_transform(full)
        # binary whole matrix
        non_zero_index = tf_idf_matrix.nonzero()
        row = non_zero_index[0]
        col = non_zero_index[1]
        lens = len(non_zero_index[0])
        data = np.array([1] * lens)
        tf_idf_matrix_BINARY = scipy.sparse.csr_matrix((data, (row, col)), shape=tf_idf_matrix.shape)
        # original up/down
        tf_idf_matrix_up = tf_idf_matrix[0:len(up), ]
        tf_idf_matrix_down = tf_idf_matrix[len(up):len(full), ]
        # original POW up/down  (*)
        tf_idf_matrix_up_POW = tf_idf_matrix_up.power(2)
        tf_idf_matrix_down_POW = tf_idf_matrix_down.power(2)
        # binary up/down
        tf_idf_matrix_BINARY_up = tf_idf_matrix_BINARY[0:len(up), ]
        tf_idf_matrix_BINARY_down = tf_idf_matrix_BINARY[len(up):len(full), ]
        # binary up/down transportation (*)
        tf_idf_matrix_BINARY_up_T = tf_idf_matrix_BINARY_up.T
        tf_idf_matrix_BINARY_down_T = tf_idf_matrix_BINARY_down.T
        del tf_idf_matrix_BINARY
        del tf_idf_matrix_BINARY_up
        del tf_idf_matrix_BINARY_down
        del tf_idf_matrix_up
        del tf_idf_matrix_down
        del tf_idf_matrix
        del vectorizer
        del non_zero_index
        # up vs down.T
        len_up = len(up)
        for c in range(0, len_up, pace):
            if len_up - c < pace:
                bound = len_up - c
            else:
                bound = pace

            mat0 = tf_idf_matrix_up_POW[c:c + bound, :] * tf_idf_matrix_BINARY_down_T
            mat1 = np.array(mat0.todense())
            for i in range(0, bound):
                mat_line = mat1[i]
                top_3_match = top_k_hybrid(mat_line, 3)
                for n in range(0, 3):
                    mj_score = mat_line[top_3_match[n]]
                    if mj_score > 0.80:
                        result_up.append(c + i)
                        result_down.append(top_3_match[n])
                        scores.append('MJS' + str(mj_score))
        # down vs up.T
        len_down = len(down)
        for c in range(0, len_down, pace):
            if len_down - c < pace:
                bound = len_down - c
            else:
                bound = pace

            mat0 = tf_idf_matrix_down_POW[c:c + bound, :] * tf_idf_matrix_BINARY_up_T
            mat1 = np.array(mat0.todense())
            for i in range(0, bound):
                mat_line = mat1[i]
                top_3_match = top_k_hybrid(mat_line, 3)
                for n in range(0, 3):
                    mj_score = mat_line[top_3_match[n]]
                    if mj_score > 0.80:
                        result_down.append(c + i)
                        result_up.append(top_3_match[n])
                        scores.append('MJS' + str(mj_score))

        del tf_idf_matrix_BINARY_up_T
        del tf_idf_matrix_down_POW
        del tf_idf_matrix_BINARY_down_T
        del tf_idf_matrix_up_POW
    else:
        pass
    # print(result_up)

    if style in ['Default','CSS']:
        # normalization of original sparse matrix
        vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams)
        tf_idf_matrix = vectorizer.fit_transform(full)
        tf_idf_matrix = pp.normalize(tf_idf_matrix.tocsc(), axis=1)
        tf_idf_matrix_up = tf_idf_matrix[0:len(up), ]
        tf_idf_matrix_down_T = tf_idf_matrix[len(up):len(full), ].T
        # tf_idf_matrix_down_T = tf_idf_matrix_down.T
        del tf_idf_matrix
        del vectorizer
        #  1. Fuzzy Matching    -  cosine similarity   CSS
        len_up = len(up)
        for c in range(0, len_up, pace):
            if len_up - c < pace:
                bound = len_up - c
            else:
                bound = pace

            mat0 = tf_idf_matrix_up[c:c + bound, :] * tf_idf_matrix_down_T  # slice_of_1000_mat
            mat1 = np.array(mat0.todense())

            for i in range(0, bound):  # bound is usually 1000
                mat_line = mat1[i]
                top_3_match = top_k_hybrid(mat_line, 3)  # top_N_match N=3

                for n in range(0, 3):  # top n , top 3
                    cs_score = mat_line[top_3_match[n]]
                    if cs_score > 0.67:
                        result_up.append(i + c)
                        result_down.append(top_3_match[n])
                        scores.append('CSS' + str(cs_score))
    else:
        pass

    # J -winkler score
    jw_scores = []
    for i in range(0, len(result_up)):
        jw_scores.append(
            distance.get_jaro_distance(up[result_up[i]], down[result_down[i]], winkler=True, scaling=0.1)
        )

    #prepare to output
    result_up_out=[ up_slice1[i] for i in  result_up  ]
    result_down_out=[down_slice1[i] for i in result_down  ]


    result_df = pd.DataFrame(list(zip(result_up_out, result_down_out,scores,jw_scores)),
                      columns=[up_col_name+'_up', down_col_name+'_down','scores','jw_scores'])
    return result_df

def clean(string):
    string1 = re.sub(r'[^a-z0-9 ]', r'', string.lower())
    string2 = re.sub('\s+', ' ', string1).strip()
    return string2


def ngrams(string, n=3):
    if len(string) < 3:
        x = []
        x.append(string + ' ' * (3 - len(string)))
        return x
    else:
        ngrams = zip(*[string[i:] for i in range(n)])
        return [''.join(ngram) for ngram in ngrams]


def whole_token(string):
    return string.split(' ')


def top_k_hybrid(a, k):
    b = np.argpartition(-a, k)[:k]
    return b[np.argsort(-a[b])]

def remove_number(token_li):
    return [i for i in token_li if i.isnumeric() == False]
def importance(frequency_li):
    importance = [1 / i for i in frequency_li]
    total = sum(importance)
    return [j / total for j in importance]
def token_analysis(up, down, up_val, down_val):  # up=up_tokens up_val=up_token_importance
    mandatory_up = up[up_val.index(max(up_val))]  # most important token - inverse importance !!!
    mandatory_down = down[down_val.index(max(down_val))]

    up2 = set(up)
    down2 = set(down)
    if up2.issubset(down2) or down2.issubset(up2):
        return 'pass'  # fully contain test
    else:
        overlap = list(up2.intersection(down2))  # overlap list
        if (mandatory_up in overlap) or (mandatory_down in overlap):
            return 'pass'
        else:
            overlap_up_sig = sum([up_val[up.index(i)] for i in overlap]) / 1
            overlap_down_sig = sum([down_val[down.index(i)] for i in overlap]) / 1
            if overlap_up_sig > 0.80 and overlap_down_sig > 0.80:
                return 'pass'
            else:
                return 'reject'
