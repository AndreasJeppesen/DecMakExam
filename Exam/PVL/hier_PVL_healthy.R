install.packages("pacman")
pacman::p_load(R2jags, parallel)

set.seed(1983)

# defining a function for calculating the maximum of the posterior density (not exactly the same as the mode)
MPD <- function(x) {
  density(x)$x[which(density(x)$y==max(density(x)$y))]
}

setwd("/work/Exam")

#----------getting the data
# data from this paper: https://www.frontiersin.org/articles/10.3389/fpsyg.2014.00849/full
# available here: https://figshare.com/articles/dataset/IGT_raw_data_Ahn_et_al_2014_Frontiers_in_Psychology/1101324

#load control data
ctr_data <- read.table("data/IGTdata_healthy_control.txt",header=TRUE)

#----------prepare data for jags models - want trial x subject arrays for choice and gain & loss ----
# identify and count unique subject IDs
subIDs <- unique(ctr_data$subjID)
nsubs <- length(subIDs)
ntrials_max <- 100

# all choices (x) and outcomes (X)
x_raw <- ctr_data$deck
X_raw <- ctr_data$gain + ctr_data$loss #note the sign!

#--- assign choices and outcomes in trial x sub matrix

#different number of trials across subjects. We'll need to fix this by padding arrays of < 100
#this is just so we can make the array
#then we'll also need to record number of valid trials for each sub, 
#then run the JAGS model on only valid trials

# empty arrays to fill
ntrials_all <- array(0,c(nsubs))
x_all <- array(0,c(nsubs,ntrials_max))
X_all <- array(0,c(nsubs,ntrials_max))

for (s in 1:nsubs) {
  
  #record n trials for subject s
  ntrials_all[s] <- length(x_raw[ctr_data$subjID==subIDs[s]])
  
  #pad trials with NA if n trials < maximum (i.e. 100)
  x_sub <- x_raw[ctr_data$subjID==subIDs[s]] 
  length(x_sub) <- ntrials_max
  
  X_sub <- X_raw[ctr_data$subjID==subIDs[s]] 
  length(X_sub) <- ntrials_max
  
  # assign arrays
  x_all[s,] <- x_sub
  X_all[s,] <- X_sub
  
}


#----------testing our data curation by running JAGS on one subject

# Now we'll fit one subject just to make sure everything works

x <- x_all[1,]
X <- X_all[1,]

ntrials <- ntrials_all[1]

# set up jags and run jags model on one subject
data <- list("x","X","ntrials") 
params<-c("w","A","theta","a","p")
temp_samples <- jags.parallel(data, inits=NULL, params,
                model.file ="PVL/PVL.txt",
                n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4)

# let's look at the posteriors for the parameters
par(mfrow=c(2,2))
plot(density(temp_samples$BUGSoutput$sims.list$w))
plot(density(temp_samples$BUGSoutput$sims.list$A))
plot(density(temp_samples$BUGSoutput$sims.list$theta))
plot(density(temp_samples$BUGSoutput$sims.list$a))

# Question: how would you expect the data to look on the basis of these posteriors?



###########################################################
#---------- run the hierarchical model on controls --------
###########################################################

x <- x_all
X <- X_all

ntrials <- ntrials_all

# set up jags and run jags model
data <- list("x","X","ntrials","nsubs") 
params<-c("mu_w","mu_A","mu_theta","mu_a","lambda_w","lambda_A","lambda_theta","lambda_a")

start_time = Sys.time()
samples <- jags.parallel(data, inits=NULL, params,
                model.file ="PVL/hier_PVL.txt",
                n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4)
end_time = Sys.time()
end_time - start_time

save(samples, file = "PVL/PVL_jags_fit.RData")
load("PVL/PVL_jags_fit.RData")


# let's look at the posteriors for the parameters
par(mfrow=c(2,2))
plot(density(samples$BUGSoutput$sims.list$mu_w), main = "mu w")
plot(density(samples$BUGSoutput$sims.list$mu_A), main = "mu A")
plot(density(samples$BUGSoutput$sims.list$mu_theta), main = "mu theta")
plot(density(samples$BUGSoutput$sims.list$mu_a), main = "mu a")

# let's look at the posteriors for the parameters
par(mfrow=c(2,2))
plot(density(samples$BUGSoutput$sims.list$lambda_w), main = "lambda w")
plot(density(samples$BUGSoutput$sims.list$lambda_A), main = "lambda A")
plot(density(samples$BUGSoutput$sims.list$lambda_theta), main = "lambda theta")
plot(density(samples$BUGSoutput$sims.list$lambda_a), main = "lambda a")

samples
traceplot(samples)



###########################################################
#-------- Get descriptive adequacy (model accuracy) -------
###########################################################

model_chosen_deck <- array(0, c(nsubs, ntrials_max))
accuracies <- array(0, dim = nsubs)

for (s in 1:nsubs) {
  
  # fit the model
  
  x <- x_all[s,]
  X <- X_all[s,]
  
  ntrials <- ntrials_all[s]
  
  data <- list("x","X","ntrials") 
  params<-c("w","A","theta","a","p")
  temp_samples <- jags.parallel(data, inits=NULL, params,
                                model.file ="PVL/PVL.txt",
                                n.chains=3, n.iter=5000, n.burnin=1000, n.thin=1, n.cluster=4)
  
  p_samples <- temp_samples$BUGSoutput$sims.list$p
  
  # empty array to fill out with maximum density p's 
  max_ps <- array(NA, c(ntrials, 4))
  
  for (t in 1:ntrials) {
    
    for (d in 1:4) {
      
      max_den_p <- MPD(p_samples[ ,t,d])
      max_ps[t,d] <- max_den_p
      
    }
    
    choice <- which.max(max_ps[t, ]) #which deck has the highest p for this trial?
    model_chosen_deck[s,t] <- choice
  }
  
  # Compare to actual data
  accuracy <- sum(model_chosen_deck[s,1:ntrials] == x[1:ntrials])
  accuracies[s] <- accuracy
  print(s)
}

for (s in 1:nsubs){
  accuracies[s] <- accuracies[s] / ntrials_all[s]
}

print(mean(accuracies))
mean_acc <- mean(accuracies)
SE_acc <- sd(accuracies, na.rm = T) / sqrt(length(accuracies))
print(SE_acc)
SD_acc <- sd(accuracies)
print(SD_acc)

acc_df <- data.frame("accuracy" = accuracies, "sub" = 1:48)

# We plot with 2.58 standard errors to either side; 99% confidence interval
acc_plot <- acc_df %>% 
  ggplot(aes(x=sub, y = accuracy)) +
  geom_point() +
  geom_hline(yintercept=0.25, linetype = "dashed") +
  geom_hline(yintercept = mean_acc, color = "darkgreen") +
  geom_rect(aes(xmin = 0, xmax=Inf, ymin = mean_acc - 2.58*SE_acc, ymax = mean_acc + 2.58*SE_acc), fill = "yellow", alpha = 0.01) +
  xlab("Subject") +
  ylab("Accuracy") +
  ggtitle("PVL-Delta Predictive accuracies") +
  theme_minimal()
acc_plot
