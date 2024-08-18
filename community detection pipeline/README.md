# Community detection pipeline procedure
The procedure is separated into 3 stages: (1) a broad gamma search, (2) fine gamma search, and (3) post-processing. 

### (1) Broad gamma search 
The broad gamma search process is found in `OAM_community_detection_broad_gamma_search.ipynb`. This is where we find 
a range of gamma value that satisfies our biological and computational criteria. 

### (2) Fine gamma search 
After identifying the best range of gamma value for each individual connectome, we performed the finner search with 
gamma increments of $0.001$ on ARC. The ARC procedure is can be found in the `arc` subfolder. 

### (3) Post processing
In the last step, we verified the output of the ARC results to determine the optimal gamma value per connectome. 
The partitions associated with the optimal gamma value is then used for analysis. 
