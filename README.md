## GraFC2T2

A general graph-based framework for top-N recommendation using content, 
temporal and trust information.

It easily combines and compares various kinds of side information for top-N recommendation. 
It encodes content-based features, temporal and trust information into a complex graph, 
and uses personalized PageRank on this graph to perform recommendation.



## Requirements

    python          2.7.14  https://www.python.org/downloads/release/python-2714/
    numpy           1.13.3  http://www.numpy.org/
    networkx        2.0     https://networkx.github.io/documentation/stable/install.html
    scikit-learn    0.19.1  https://sklearn.org/
    scipy           0.19.1  https://libraries.io/pypi/scipy/0.19.1
    


## Datasets

epinions-data.txt, epinions-trust-network.txt, ciao-data.txt and ciao-trust-network
are extracted from data published by Dr. Jiliang Tang 
( https://www.cse.msu.edu/~tangjili/trust.html )

Each row of epinions-data.txt and ciao-data.txt files contains the following information:
user_id; item_id; rating; content_id; timestamp

Each row of epinions-trust-network.txt and ciao-trust-network.txt files contains the following information:
user_id; trusted_user_id

Epinions data statistics
------------------------
    start date: 2010-01-01
    End date:   2010-12-31
    Users:      1999
    Items:      24861
    Contents:   24
    ratings:    28399
    trust:      5529

Ciao data statistics
--------------------
    start date: 2007-01-01
    End date:   2010-12-31
    Users:      890
    Items:      9084
    Contents:   6
    ratings:    12753
    trust:      23398



## Evaluation

Several evaluation metrics and several values of Top-N are available.

prec: Precision, recall: Recall, map: Mean average precision, hr: Hit ratio, mrr: Mean reciprocal rank

N in {1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100}

Each evaluation metric is of the form "metric_name@N" 

examples: "prec@10", "recall@30", "map@20", "hr@15", "mrr@50"



## Deployment

Put the framework.py and launch.py ​​files in the same folder.

Open the launch.py ​​file and configure as following:

1- Path to the dataset

    # File that contains link stream data {(u, i, r, c, t)}
    input_dataset_file = "ciao-data.txt" 

2- Separator used in the dataset file

    separator_df = ";"   # separator of elements in a row of the dataset file

3- Position of user, item, rating, content-based feature and timestamp in a row dataset

    pos_u = 0   # position of user u in a row [the first position is 0 (zero)]
    pos_i = 1   # position of item i 
    pos_r = 2   # position of rating r 
    pos_c = 3   # position of content-based feature c 
    pos_t = 4   # position of timestamp t

4- Path to the trust network file

    # File that contains explicit trust network
    # This is optional if you don't want to use explicit trust in GraFC2T2
    # Put None or "" to ignore explicit trust
    input_trust_network = "ciao-trust-network.txt"  
       
5- Separator used in the trust network file   

    separator_tn = ";"   # separator of elements in a row of the trust network file

6- Position of user and trusted user in a row row of trust network

    pos_user = 0        # position of user u who trust another
    pos_trusted = 1     # position of the trusted user

7- Specify the metrics of evaluations that you want

    # you can add any metric in the form prec@N (Precision), recall@N, map@N, hr@N (Hit Ratio)
    # with N in {1, 2, 3, 5, 10, 15, 20, 30, 40, 50, 100}
    eval_metric = ["prec@10", "recall@10", "map@10", "hr@10"]   
       
8- Give the number of time slice desired (it must be >= 2)

    # the time is divided in time slices of equal length
    number_of_time_slices = 8 
       
9- Configure the recommendation graphs that you want to run simultaneously
       
    """
    It is possible to configure several recommender graphs
    In this example, two recommender graphs are configurated 

    ---->  recommender graph construction

    graph : basic graph {"BIP", "STG", "LSG"}
    content : content-based method {"NA", "CI", "CIU"}
    time : time-weight function {"NA", "EDF", "LDF"}
    trust : trust method in PageRank {"NA", "ET", "IT"}

    ---->  recommender graph parameters setting

    Delta : STG session duration    {7,30,90,180,365,540,730} in days
    Beta : STG long-term preference    {0.1, 0.3, 0.5, 0.7, 0.9} 
    T0 : half life of Exponential Decay Function (EDF) and 
        Logistic Decay Function (LDF) {7,30,90,180,365,540,730} in days  
    K : decay slope of LDF    {0.1, 0.5, 1, 5, 10, 50, 100}
    Gamma : influence of trusted users    {0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9}         
    Alpha : damping factor for PageRank    {0.05, 0.1, 0.15, 0.3, 0.5, 0.7, 0.9}
    """
    rs1_param = {
        "graph":"BIP", "content":"CIU", "time":"EDF", "trust":"IT", 
        "delta":0, "beta":0, "t0":180, "k":0, "gamma":0.9, "alpha":0.9
    }

    rs2_param = {
        "graph":"STG", "content":"CIU", "time":"LDF", "trust":"IT", 
        "delta":540, "beta":0.1, "t0":365, "k":10, "gamma":0.7, "alpha":0.9
    }

    rs_list = [rs1_param, rs2_param] # list of configurated recsys 
       
10- Start the execution

    save change and close the file launch.py 
    then run the command  " python launch.py "

The results file is in the same folder as the framework.py and launch.py ​​files, 
and has a name of the form "GraFC2T2-RESULTS-1547812182.txt" 
where 1547812182 is the creation timestamp of the file.       

For the example described in this readme, the results file contains :

    ----> STG-CIU-LDF-IT-Alpha-0.9-Delta-540-Beta-0.1-T0-365-K-10-Gamma-0.7
    prec@10 = 0.0127272727273
    recall@10 = 0.0566801619433
    map@10 = 0.0333643578644
    hr@10 = 0.110909090909

    ----> BIP-CIU-EDF-IT-Alpha-0.9-T0-180-Gamma-0.9
    prec@10 = 0.012
    recall@10 = 0.0534412955466
    map@10 = 0.0332146464646
    hr@10 = 0.101818181818



## Citation

If you use this code for your paper, please cite :

    The link: 
    "https://github.com/nzekonarmel/GraFC2T2"

    The paper :
    @inproceedings{nzeko2017time,
      title={Time Weight Content-Based Extensions of Temporal Graphs for Personalized Recommendation},
      author={Nzekon Nzeko'o, Armel Jacques and Tchuente, Maurice and Latapy, Matthieu},
      booktitle={WEBIST 2017-13th International Conference on Web Information Systems and Technologies},
      year={2017}
    }

