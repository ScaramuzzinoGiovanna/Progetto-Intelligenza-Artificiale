
# coding: utf-8

# In[1]:


import pandas as pd
col_names = ["duration","protocol_type","service","flag","src_bytes",
    "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
    "logged_in","num_compromised","root_shell","su_attempted","num_root",
    "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
    "is_host_login","is_guest_login","count","srv_count","serror_rate",
    "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
    "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
    "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
    "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]

data_file=pd.read_csv('C:\\Users\\Scara\\Documents\\Intelligenza_Artificiale\\PROGETTO IA\\kddcup.data_10_percent_corrected', header=None, names = col_names)
kdd_data_corrected = pd.read_csv('C:\\Users\\Scara\\Documents\\Intelligenza_Artificiale\\PROGETTO IA\\corrected', header=None, names = col_names )


# In[2]:


protocol_types = {"icmp":0, "tcp":1, "udp":2}

services = {"auth":0, "bgp":1, "courier":2, "csnet_ns":3, "ctf":4, "daytime":5, "discard":6, "domain":7, "domain_u":8, "echo":9,
            "eco_i":10, "ecr_i":11, "efs":12, "exec":13, "finger":14, "ftp":15, "ftp_data":16, "gopher":17, "hostnames":18,
            "http":19, "http_443":20, "imap4":21, "IRC":22, "iso_tsap":23, "klogin":24, "kshell":25, "ldap":26, "link":27,
            "login":28, "mtp":29, "name":30, "netbios_dgm":31, "netbios_ns":32, "netbios_ssn":33, "netstat":34, "nnsp":35,
            "nntp":36, "ntp_u":37, "other":38, "pm_dump":39, "pop_2":40, "pop_3":41, "printer":42, "private":43, "red_i":44,
            "remote_job":45, "rje":46, "shell":47, "smtp":48, "sql_net":49, "ssh":50, "sunrpc":51, "supdup":52, "systat":53,
            "telnet":54, "tftp_u":55, "time":56, "tim_i":57, "urh_i":58, "urp_i":59, "uucp":60, "uucp_path":61, "vmnet":62,
            "whois":63, "X11":64, "Z39_50":65, 'icmp':66}

flags = {"OTH":0,"REJ":1,"RSTO":2,"RSTOS0":3,"RSTR":4,"S0":5,"S1":6,"S2":7,"S3":8,"SF":9,"SH":10}

labels={'normal.':0, 'snmpgetattack.':1, 'named.':2, 'xlock.' :3, 'smurf.':4, 'ipsweep.':5,'multihop.':6, 'xsnoop.':7,
        'sendmail.':8, 'guess_passwd.' :9, 'saint.':11,'buffer_overflow.':12, 'portsweep.':13, 'pod.':14, 'apache2.':15,
        'phf.':16, 'udpstorm.':17, 'warezmaster.':18, 'perl.':19, 'satan.':20, 'xterm.':21, 'mscan.':22, 'processtable.':23, 
        'ps.':24,'nmap.':25, 'rootkit.':26, 'neptune.':27, 'loadmodule.':28, 'imap.':29, 'back.':30, 'httptunnel.':31,
        'worm.':32, 'mailbomb.':33, 'ftp_write.':34, 'teardrop.':35, 'land.':36, 'sqlattack.':37,'snmpguess.':38, 'spy.':39,'warezclient.':40}


#associa le label_test alle classi
dict={ 0:'Normal', 1:'R2L', 2:'R2L', 3:'R2L', 4:'DOS', 5:'Probe', 6:'R2L', 7:'R2L',
        8:'R2L', 9:'R2L', 11:'Probe',12:'U2R', 13:'Probe', 14:'DOS', 15:'DOS',
        16:'R2L', 17:'DOS', 18:'R2L', 19:'U2R', 20:'Probe', 21:'U2R', 22:'Probe', 23:'DOS', 
        24:'U2R', 25:'Probe', 26:'U2R', 27:'DOS', 28:'U2R', 29:'R2L', 30:'DOS', 31:'U2R',
        32:'R2L', 33:'DOS', 34:'R2L', 35:'DOS', 36:'DOS', 37:'U2R',38:'R2L', 39:'R2L',40:'R2L'}

classes={'DOS':1, 'U2R':2, 'R2L':3,'Probe':4, 'Normal':5}


# In[3]:


import numpy as np
def to_matrix(dataset):
    return np.asmatrix(dataset)  


# In[4]:


kdd_data_corrected1=to_matrix(kdd_data_corrected)  
print(kdd_data_corrected1)


# In[5]:


data_file1=to_matrix(data_file)
print(data_file1)


# In[6]:


X_test=[]

for row in range(0,311029):
    X1=[]
    for col in range(0,41):
        tmp=kdd_data_corrected1[row,col]
        if col==1 :
            X1.append(protocol_types[tmp])
        elif col==2:
            X1.append(services[tmp])
        elif col==3:
            X1.append(flags[tmp])
        else:
            X1.append(tmp)
    X_test.append(X1)
#print(X_test)


# In[7]:


y_test=[]

for row in range(0,311029):
    tmp=kdd_data_corrected.loc[row]['label']
    y_test.append(labels[tmp])
    
#print(y_test)


# In[8]:


X_train=[]

for row in range(0,494021):
    X1=[]
    for col in range(0,41): 
        tmp=data_file1[row,col]
        if col==1 :
            X1.append(protocol_types[tmp])
        elif col==2:
            X1.append(services[tmp])
        elif col==3:
            X1.append(flags[tmp])
        else:
            X1.append(tmp)
    X_train.append(X1)

#print(X_train)


# In[9]:


y_train=[]

for row in range(0, 494021):
    tmp=data_file.loc[row]['label']
    y_train.append(labels[tmp])

#print(y_train)


# In[10]:


#dopo
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

classifier= DecisionTreeClassifier(criterion='entropy')
model=classifier.fit(X_train, y_train)
pred_test=classifier.predict(X_test)
pred_train=classifier.predict(X_train)

print("Accuracy of Decision Tree classifier on train set after classification:%.2f%%" % (100*(accuracy_score(y_train,pred_train))))


# In[11]:


pred1=[]

for elem in pred_test:
    tmp=dict[elem]
    pred1=np.append(pred1,classes[tmp])
    
print(pred1)


# In[12]:


y_test1=[]

for elem in y_test:
    tmp=dict[elem]
    y_test1=np.append(y_test1, classes[tmp])
    
print(y_test1)


# In[13]:


print("Accuracy of Decision Tree classifier on test set after classification:%.2f%%" % (100*accuracy_score(y_test1,pred1)))


# In[14]:


from sklearn.metrics import confusion_matrix
cm_after = confusion_matrix(y_test1, pred1, labels=[5, 1, 3, 2, 4])
print(cm_after)


# In[15]:


vect_sum=[]
for row in range(0,5):
    sum=0
    for col in range(0,5):
        sum+=cm_after[row,col]
    vect_sum.append(sum)

cm=[]
for i in range(0,5):
    x=[] 
    for j in range(0,5):
            x.append('{:.2%}.'.format(cm_after[i,j]/vect_sum[i]))
    cm.append(x)  
            
Conf_matrix_perc=to_matrix(cm)
print(Conf_matrix_perc)


# In[16]:


tot_ex=0
for i in range(0,len(vect_sum)):
    tot_ex+=vect_sum[i]
    
tot_diag=0    
for row in range(0, len(cm_after)):
    for col in range (0,len(cm_after)):
        if row==col:
            tot_diag+=cm_after[row,col]

print("Accuracy of Confusion Matrix after classification:%.2f%%" % (100*(tot_diag/tot_ex)))


# In[17]:


y_train1=[]

for elem in y_train:
    tmp=dict[elem]
    y_train1=np.append(y_train1, classes[tmp])
    
print(y_train1)


# In[18]:


#prima
classifier= DecisionTreeClassifier(criterion='entropy') 
model=classifier.fit(X_train, y_train1)
pred=classifier.predict(X_test)
pred_train=classifier.predict(X_train)


# In[19]:


print("Accuracy of Decision Tree classifier on train set before classification: %.2f%%" % (100*accuracy_score(y_train1, pred_train)))
print("Accuracy of Decision Tree classifier on test set before classification: %.2f%%" % (100*accuracy_score(y_test1, pred)))


# In[20]:


cm_before = confusion_matrix(y_test1, pred, labels=[5, 1, 3, 2, 4])
print(cm_before)


# In[21]:


vect_sum=[]
for row in range(0,len(cm_before)):
    sum=0
    for col in range(0,5):
        sum+=cm_before[row,col]
    vect_sum.append(sum)

cm=[]
for i in range(0,len(cm_before)):
    x=[] 
    for j in range(0,len(cm_before)):
            x.append('{:.2%}.'.format(cm_before[i,j]/vect_sum[i]))
    cm.append(x)  
            
Conf_matrix_perc=to_matrix(cm)
print(Conf_matrix_perc)


# In[22]:


tot_ex=0
for i in range(0,len(vect_sum)):
    tot_ex+=vect_sum[i]
    
tot_diag=0    
for row in range(0, len(cm_before)):
    for col in range (0,len(cm_before)):
        if row==col:
            tot_diag+=cm_before[row,col]

print("Accuracy of Confusion Matrix before classification :%.2f%%" % (100*(tot_diag/tot_ex)))

