
# Federated Learning Test Environment

Testing on the hypothesis that the server side idle time can be utilized for a better optimization.
While there are some research on PFL(Parallel Federated Learning), which has parallel servers that run concurrently to achieve 
a faster convergence rate.

This research however, looks towards the inner workings of the basic federated learning architecture and focuses on the idle
server times. The idleness of the servers comes from the time during each clients training and evalaution epochs. Thus, in order to maxmize the 
server aggregations we propose a *PWFL*, a Parallel Weight Federated Learning, novel approach to the any counterpart, as the server 
recognizes multiple weights and global model running on different model running concurrently. Given that many research can be done 
on a cross-silo setting(where the central servers have enough computing power), we can validate that this parallel can be achieved

During each global round, the PWFL server can# PWFL
