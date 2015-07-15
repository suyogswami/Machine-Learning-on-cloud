##############################################################################
##  Name: Suyog S Swami
##  ID: 1001119101
##  Course: Cloud Computing (CSE-6331) Batch: 1PM-3PM
##  Title: Machine Learning(Kmeans CLustering in Cloud)
##  Reference:  http://glowingpython.blogspot.com/2012/04/k-means-clustering-with-scipy.html
##              http://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.astype.html
##              http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.spatial.distance.euclidean.html
##              http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.cluster.vq.vq.html#scipy.cluster.vq.vq
##              http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans
##              http://glowingpython.blogspot.com/2012/04/k-means-clustering-with-scipy.html
##              http://52.11.30.186:5000/
##############################################################################
import csv
import operator
import numpy as np
from scipy.cluster.vq import kmeans,vq
from scipy.spatial.distance import euclidean
from flask import Flask,render_template,request,url_for
import matplotlib.pyplot as plt
#import matplotlib as mp

app = Flask(__name__)

@app.route('/')
def inputs():
    return render_template('input.html')

@app.route('/output/',methods=['GET','POST'])
def output():
    x_coordinate=request.form.get('x_coordinate')
    y_coordinate=request.form.get('y_coordinate')
    n_clusters=int(request.form.get('n_clusters'))

    column_ind_nm={1:'time', 2:'latitude', 3:'longitude', 4:'depth', 5:'mag', 6:'magType', 7:'nst', 8:'gap', 9:'dmin', 10:'rms', 11:'net', 12:'id', 13:'updated', 14:'place', 15:'type'}
    #n_clusters=input('Input the number of clusters: ')
    
    with open('all_month.csv','r') as csvf:
        reader=csv.reader(csvf,delimiter=',')
        header=[]
        refined_data=[]
        i=0
        for r in reader:
            if i==0:
                header=r
        
            else:
                refined_data.append([(r[header.index(column_ind_nm.get(int(x_coordinate)))]) or float(0.0),r[header.index(column_ind_nm.get(int(y_coordinate)))] or int(0)])
            i=i+1
        with open('result.csv','w',newline='') as result:
            spamwriter=csv.writer(result,delimiter=',')
            for r in refined_data:
                spamwriter.writerow([r[0], r[1]])
                
        refined_data=np.array(refined_data)
        refined_data=refined_data.astype(float)
        #print refined_data
        centroids,distort = kmeans(refined_data,n_clusters)
        centroids_list=centroids.tolist()
        print('\nCENTROIDS:\n')
        for c_l in centroids_list:    
            print(c_l)

        idx,_=vq(refined_data,centroids)
        #print idx
        euclid_dict={}    
        print('\nDistance Between Centroids \n')
        for c in centroids_list:
            for ce in centroids_list[centroids_list.index(c)+1:]:
                euclid_dict[str(centroids_list.index(c))+' and '+str(centroids_list.index(ce))]=euclidean(c,ce)
        euclid_dict=sorted(euclid_dict.items(),key=operator.itemgetter(0))
        print(euclid_dict)
        cnt_idx=idx.tolist()
        dict_index={}
        print('\nNumber of points in each cluster\n')
        for i in cnt_idx:
            dict_index[i]=cnt_idx.count(i)
        print(dict_index)
        with open('forbarchart.csv','w',newline='') as barchart:
            forbarwriter=csv.writer(barchart,delimiter=',')
            forbarwriter.writerow(['Cluster','Point'])
            for k,v in dict_index.items():
                forbarwriter.writerow([k, v])
        color_array=["r.", "g.", "b.","y.","k.","b.","m.","c."]
        for n in range(n_clusters):
            plt.plot(refined_data[idx==n,0],refined_data[idx==n,1],color_array[n%8],marker="x", markersize=5)
        plt.plot(centroids[:,0],centroids[:,1], "sm", markersize=5)    
        plt.savefig("static\image.jpg")
        plt.clf()
        #fig=plt.figure()
        #x=plt.subplot(111)
        #for m,o in dict_index.items():
        print(len(dict_index))
        print(dict_index.values())
        plt.bar(range(len(dict_index)),dict_index.values(),width=1/1.5,color="blue")
        plt.xticks(range(len(dict_index)), list(dict_index.keys()))
        plt.savefig("static\image_bar.jpg")
        
    return render_template("output.html",dict_index=dict_index,centroids_list=centroids_list)
    
if __name__ == "__main__":
    # Setting debug to True enables debug output. This line should be
    # removed before deploying a production app.
    app.debug = True
    app.run("0.0.0.0")
